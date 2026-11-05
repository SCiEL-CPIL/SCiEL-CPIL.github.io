"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Bayesian Linear Bandit with Thompson Sampling.
    
    Uses a Gaussian posterior over arm weights with an adaptive discount factor
    that responds to prediction errors (Mahalanobis distance of residuals),
    allowing faster adaptation when the environment changes.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t_start = time.time()
        self.deadline = 150.0  # wall-clock limit

        # Sufficient statistics for each arm
        # A[a] is the precision matrix (inverse covariance)
        # b[a] is the sufficient statistic for the mean
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Track prediction errors for adaptive discounting
        self.error_history = [0.0 for _ in range(n_arms)]
        self.base_discount = 0.995
        self.min_discount = 0.95
        self.max_discount = 0.999

        # Cache for inverse matrices to avoid repeated inversions
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self._cache_valid = [True] * n_arms

    def _get_theta_and_cov(self, arm):
        """Get mean (theta) and covariance for a given arm."""
        if not self._cache_valid[arm]:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self._cache_valid[arm] = True
        theta = self.A_inv[arm] @ self.b[arm]
        cov = self.A_inv[arm]
        return theta, cov

    def select(self, context):
        """Select an arm using Thompson Sampling."""
        context = np.asarray(context, dtype=np.float64)

        # Check deadline
        if time.time() - self.t_start > self.deadline - 5.0:
            # Safety: just return the arm with highest mean estimate
            means = []
            for a in range(self.n_arms):
                theta, _ = self._get_theta_and_cov(a)
                means.append(theta @ context)
            return int(np.argmax(means))

        scores = []
        for a in range(self.n_arms):
            theta, cov = self._get_theta_and_cov(a)
            # Sample from posterior: theta ~ N(theta_hat, cov)
            sample = theta + np.random.multivariate_normal(np.zeros(self.dim), cov, size=1)[0]
            scores.append(sample @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the posterior with a new observation."""
        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error (residual)
        theta, cov = self._get_theta_and_cov(arm)
        pred = theta @ context
        residual = reward - pred

        # Compute Mahalanobis distance of the residual
        # Var(reward | context) = context^T cov context + sigma^2
        # For simplicity, use the Mahalanobis distance of the residual
        # relative to the predicted variance
        var_pred = context @ cov @ context
        if var_pred > 1e-10:
            mahalanobis = abs(residual) / np.sqrt(var_pred + 1e-10)
        else:
            mahalanobis = abs(residual)

        # Update error history (exponential moving average)
        self.error_history[arm] = 0.9 * self.error_history[arm] + 0.1 * mahalanobis

        # Adaptive discount factor: higher error -> lower discount (faster forgetting)
        # Map error to discount factor
        error = self.error_history[arm]
        if error < 1.0:
            discount = self.max_discount
        elif error > 5.0:
            discount = self.min_discount
        else:
            # Linear interpolation
            discount = self.max_discount - (self.max_discount - self.min_discount) * (error - 1.0) / 4.0
        discount = np.clip(discount, self.min_discount, self.max_discount)

        # Update sufficient statistics with discounting
        self.A[arm] = discount * self.A[arm] + np.outer(context, context)
        self.b[arm] = discount * self.b[arm] + reward * context

        # Invalidate cache
        self._cache_valid[arm] = False
# EVOLVE-BLOCK-END
