"""Linear Thompson Sampling with forgetting factor for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices (A) and mean vectors (b)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Store inverse A for faster sampling
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        # Forgetting factor: balance between memory and adaptability
        self.alpha = 0.99
        # Assumed noise variance
        self.sigma_sq = 1.0
        # Regularization to keep A invertible
        self.reg = 1e-3
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin before 120s
        self._rng = np.random.default_rng()

    def _compute_posterior(self, a):
        """Compute posterior mean and covariance for arm a."""
        A = self.A[a]
        b = self.b[a]
        # Ensure positive definiteness
        A_reg = A + self.reg * np.eye(self.d)
        try:
            A_inv = np.linalg.inv(A_reg)
        except np.linalg.LinAlgError:
            A_inv = self.A_inv[a]
        self.A_inv[a] = A_inv
        theta_mean = A_inv @ b
        return theta_mean, A_inv

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: pick the arm with highest estimated mean (no exploration)
            scores = []
            for a in range(self.n):
                theta_mean, _ = self._compute_posterior(a)
                scores.append(theta_mean @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta_mean, A_inv = self._compute_posterior(a)
            # Sample from posterior: theta ~ N(theta_mean, A_inv)
            # Use Cholesky for sampling
            try:
                L = np.linalg.cholesky(A_inv)
                z = self._rng.standard_normal(self.d)
                theta_sample = theta_mean + L @ z
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                theta_sample = theta_mean
            scores.append(theta_sample @ context)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        # Forgetting: decay old information
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
