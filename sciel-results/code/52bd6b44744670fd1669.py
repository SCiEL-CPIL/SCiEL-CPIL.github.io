"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices A and mean vectors b
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor: 0.99 provides a balance between memory and adaptability
        self.alpha = 0.99
        # Assumed noise variance for Thompson Sampling
        self.sigma_sq = 1.0

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Compute the posterior mean theta_hat
            inv_A = np.linalg.inv(self.A[a])
            theta_hat = inv_A @ self.b[a]
            
            # Compute the posterior covariance V = sigma^2 * A^-1
            V = self.sigma_sq * inv_A
            
            # Sample theta from N(theta_hat, V)
            # We can sample by: theta = theta_hat + L @ z, where L is Cholesky of V and z ~ N(0, I)
            # However, V might be singular or ill-conditioned if A is not full rank initially, 
            # but A starts as I and we add outer products, so it should be positive definite.
            try:
                L = np.linalg.cholesky(V)
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (shouldn't happen often with regularization)
                # Use eigen decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(V)
                # Clip small negative eigenvalues to 0
                eigenvalues = np.clip(eigenvalues, 0, None)
                L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z

            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Apply forgetting factor and update with new observation
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
