function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X= [ones(m,1) X]; 12x2
%theta; 2x1
%y; %12x1
h= X*theta; %12x1
J= (1/(2*m))* sum((h-y).*(h-y));
reg= (lambda/(2*m))*sum(theta(2:end).*theta(2:end));
J= J + reg;

%grad(1)= (1/m)* sum((h-y).* X(:,1));
%grad(2:end)= (lambda/m)*theta(2:end) + (1/m)*sum(X(:,2:end)' * (h-y));


G = (lambda/m) .* theta;
G(1) = 0; % this is always 0

grad = ((1/m) .* X' * (X*theta - y)) + G;











% =========================================================================

grad = grad(:);

end
