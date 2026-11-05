# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Discounted Thompson Sampling for linear bandits."""

    def __init__(self, n_arms, dim, alpha=0.95, sigma2=1.0, ridge=1e-2):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension (intercept + context)
        self.alpha = alpha
        self.sigma2 = sigma2
        self.ridge = ridge
        
        # Per-arm statistics for Bayesian Linear Regression
        # V: Precision matrix (inverse covariance)
        # w: Weighted sum of x * y
        # We use the update: V_t = alpha * V_{t-1} + (1/sigma2) * x x^T + ridge * I
        # w_t = alpha * w_{t-1} + (1/sigma2) * x * y
        
        # Initialize with a weak prior (high variance, low precision)
        # But to ensure stability, we start with a moderate precision.
        # If we start with I, variance is 1.
        self.V = [np.eye(self.D) * self.ridge for _ in range(n_arms)]
        self.w = [np.zeros(self.D) for _ in range(n_arms)]
        
        self.t = 0

    def _augment(self, context):
        context = np.asarray(context, dtype=float).ravel()
        x_aug = np.zeros(self.D)
        x_aug[0] = 1.0
        x_aug[1:] = context
        return x_aug

    def select(self, context):
        x_aug = self._augment(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            try:
                V_inv = np.linalg.inv(self.V[a])
            except np.linalg.LinAlgError:
                V_inv = np.linalg.pinv(self.V[a])
            
            # Posterior mean
            theta_mean = V_inv @ self.w[a]
            
            # Posterior covariance: Sigma = sigma2 * V_inv
            # We sample theta ~ N(theta_mean, sigma2 * V_inv)
            # To sample efficiently, we can use Cholesky or just multivariate_normal
            # For small D, multivariate_normal is fine.
            cov = self.sigma2 * V_inv
            
            # Ensure symmetry and positive definiteness for sampling
            # np.random.multivariate_normal can be slow, so we do manual sampling
            # theta = theta_mean + L @ z, where L is Cholesky of cov
            try:
                L = np.linalg.cholesky(cov)
                z = np.random.randn(self.D)
                theta_sample = theta_mean + L @ z
            except np.linalg.LinAlgError:
                # Fallback to mean if covariance is not PD
                theta_sample = theta_mean
            
            scores[a] = theta_sample @ x_aug
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Discounted update
        self.V[arm] *= self.alpha
        self.w[arm] *= self.alpha
        
        # Add new observation
        # Note: We add ridge to V to maintain positive definiteness and prevent drift to zero
        self.V[arm] += np.outer(x_aug, x_aug) / self.sigma2
        self.V[arm] += np.eye(self.D) * self.ridge
        self.w[arm] += x_aug * reward / self.sigma2
        
        self.t += 1
# EVOLVE-BLOCK-END
