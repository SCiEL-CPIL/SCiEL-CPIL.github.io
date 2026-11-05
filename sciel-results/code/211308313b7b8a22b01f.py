import numpy as np

class Policy:
    """Linear Thompson Sampling with Exponential Forgetting for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Forgetting factor
        self.lambda_reg = 1.0  # Initial regularization (precision)
        
        # Sufficient statistics: S = sum(x x^T), b = sum(x r)
        # Initialized with prior precision lambda_reg * I
        self.S = [self.lambda_reg * np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        
        # Cache for inverse S to avoid repeated inversions if possible, 
        # but since S changes every update, we might just invert in select.
        # For small d, inversion is cheap.

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Compute posterior mean and covariance
            # V = S^{-1}, mu = V b
            try:
                V = np.linalg.inv(self.S[a])
                mu = np.dot(V, self.b[a])
            except np.linalg.LinAlgError:
                # Fallback if singular
                V = np.eye(self.d)
                mu = np.zeros(self.d)
            
            # Sample theta from N(mu, V)
            # We can use np.random.multivariate_normal, but it's slow.
            # Use Cholesky or SVD for sampling.
            # V is symmetric positive definite.
            L = np.linalg.cholesky(V + 1e-6 * np.eye(self.d)) # Add small jitter for stability
            z = np.random.randn(self.d)
            theta = mu + np.dot(L, z)
            
            # Compute score
            score = np.dot(theta, context)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update sufficient statistics with forgetting
        self.S[arm] = self.gamma * self.S[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + context * reward
