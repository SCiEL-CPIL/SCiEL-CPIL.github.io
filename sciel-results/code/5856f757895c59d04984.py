# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Discounted Bayesian Linear Bandit with Thompson Sampling."""

    def __init__(self, n_arms, dim, alpha=0.95, sigma2=1.0, prior_precision=1.0):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension (intercept + context)
        self.alpha = alpha
        self.sigma2 = sigma2
        self.prior_precision = prior_precision
        
        # Per-arm posterior parameters
        # P: Precision matrix (inverse covariance)
        # v: P @ mu (weighted sum of x*y scaled by 1/sigma2)
        self.P = [np.eye(self.D) * prior_precision for _ in range(n_arms)]
        self.v = [np.zeros(self.D) for _ in range(n_arms)]
        
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
            # Compute mean and covariance
            # mu = P^{-1} v
            # Cov = P^{-1}
            # Add small jitter for numerical stability
            P_reg = self.P[a] + 1e-6 * np.eye(self.D)
            
            try:
                P_inv = np.linalg.inv(P_reg)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P_reg)
            
            mu = P_inv @ self.v[a]
            cov = P_inv
            
            # Sample theta from N(mu, cov)
            # Ensure cov is symmetric for multivariate_normal
            cov = (cov + cov.T) / 2.0
            
            # Use standard normal sampling and transform for speed/stability
            # theta = mu + L @ z, where L L^T = cov
            try:
                L = np.linalg.cholesky(cov)
                z = np.random.randn(self.D)
                theta = mu + L @ z
            except np.linalg.LinAlgError:
                # Fallback to direct sampling if Cholesky fails
                theta = np.random.multivariate_normal(mu, cov)
            
            scores[a] = theta @ x_aug
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Discounted Bayesian Update
        # P_new = alpha * P_old + (1/sigma2) * x x^T
        # v_new = alpha * v_old + (1/sigma2) * x y
        
        self.P[arm] = self.alpha * self.P[arm] + (1.0 / self.sigma2) * np.outer(x_aug, x_aug)
        self.v[arm] = self.alpha * self.v[arm] + (1.0 / self.sigma2) * x_aug * reward
        
        self.t += 1
# EVOLVE-BLOCK-END
