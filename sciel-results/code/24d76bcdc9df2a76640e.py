import numpy as np

class Policy:
    """Recursive Least Squares (RLS) with Exponential Forgetting for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lambda_reg = 0.1  # Initial regularization (inverse of initial A_inv scale)
        self.forgetting = 0.95  # Exponential forgetting factor
        self.c_alpha = 1.5     # Exploration bonus coefficient
        
        # Initialize theta (parameters) and A_inv (inverse covariance) for each arm
        # A_inv starts as (1/lambda) * I, so theta is 0
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]
        self.A_inv = [(1.0 / self.lambda_reg) * np.eye(self.d) for _ in range(n_arms)]

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Predicted mean
            pred = np.dot(self.theta[a], context)
            
            # Variance term: sqrt(x^T A_inv x)
            # We compute x^T A_inv first, then dot with x for efficiency
            temp = np.dot(self.A_inv[a], context)
            var = np.sqrt(np.dot(context, temp))
            
            # UCB score
            scores.append(pred + self.c_alpha * var)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # RLS Update with forgetting factor
        # A_new = alpha * A_old + x x^T
        # We update A_inv using the matrix inversion lemma:
        # A_inv_new = (1/alpha) * A_inv_old - (1/alpha) * (A_inv_old @ x @ x^T @ A_inv_old) / (alpha + x^T @ A_inv_old @ x)
        
        # Optimization: Compute terms step-by-step to avoid full matrix multiplications where possible
        # Let K = A_inv_old @ x
        K = np.dot(self.A_inv[arm], context)
        
        # Denominator: alpha + x^T K
        denom = self.forgetting + np.dot(context, K)
        
        # Update A_inv
        # A_inv_new = (1/alpha) * A_inv_old - (1/(alpha * denom)) * outer(K, K)
        self.A_inv[arm] = (1.0 / self.forgetting) * self.A_inv[arm] - (1.0 / (self.forgetting * denom)) * np.outer(K, K)
        
        # Update theta
        # theta_new = theta_old + K * (reward - x^T theta_old) / denom
        pred = np.dot(self.theta[arm], context)
        error = reward - pred
        self.theta[arm] = self.theta[arm] + (K * error) / denom
