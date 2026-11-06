import numpy as np

class Policy:
    """Discounted Thompson Sampling for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lambda_reg = 0.1
        self.forgetting = 0.95
        # We use Thompson Sampling, so no UCB constant needed.
        # However, we might need a noise scale for the reward variance assumption.
        # Standard Linear TS assumes y ~ N(theta^T x, sigma^2).
        # Posterior: theta | D ~ N(theta_hat, sigma^2 A_inv)
        # We assume sigma^2 = 1 for the sampling covariance.
        
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]
        self.A_inv = [(1.0 / self.lambda_reg) * np.eye(self.d) for _ in range(n_arms)]

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Mean of posterior
            mean = self.theta[a]
            # Covariance of posterior
            # If we assume unit variance for rewards, Cov = A_inv
            # If we want to be safe, we can scale it. 
            # Standard TS uses Cov = A_inv.
            cov = self.A_inv[a]
            
            # Sample theta from N(mean, cov)
            # Note: np.random.multivariate_normal can be unstable if cov is singular,
            # but A_inv is positive definite due to regularization.
            sampled_theta = np.random.multivariate_normal(mean, cov)
            
            # Score is the predicted reward using the sampled parameter
            scores.append(np.dot(sampled_theta, context))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # RLS Update with forgetting factor
        K = np.dot(self.A_inv[arm], context)
        denom = self.forgetting + np.dot(context, K)
        
        self.A_inv[arm] = (1.0 / self.forgetting) * self.A_inv[arm] - (1.0 / (self.forgetting * denom)) * np.outer(K, K)
        
        pred = np.dot(self.theta[arm], context)
        error = reward - pred
        self.theta[arm] = self.theta[arm] + (K * error) / denom
