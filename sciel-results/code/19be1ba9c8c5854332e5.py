"""
Bayesian Linear Bandit with adaptive Thompson Sampling.
Uses event-driven forgetting: strong decay on regime shift, mild decay otherwise.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t_start = time.time()
        self.deadline = self.t_start + 115.0

        # Bayesian linear model per arm
        # A = sum of outer products (with forgetting), b = sum of reward * context
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Cholesky factors for efficient sampling
        self.L = [np.linalg.cholesky(self.A[a]) for a in range(n_arms)]

        # Adaptive forgetting parameters
        self.strong_decay = 0.5    # Applied on regime shift
        self.mild_decay = 0.99     # Applied normally
        self.regime_threshold = 1.0  # Mahalanobis distance threshold for regime shift

    def _update_chol(self, A):
        """Update Cholesky factor with fallback regularization."""
        try:
            return np.linalg.cholesky(A + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            A = A + 1e-4 * np.eye(self.d)
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                return np.eye(self.d)

    def _get_theta(self, arm):
        """Get point estimate theta = A^{-1} b."""
        try:
            L = self.L[arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            return theta
        except Exception:
            return np.zeros(self.d)

    def _mahalanobis_distance(self, arm, context, residual):
        """Compute Mahalanobis distance of residual under posterior uncertainty.
        d = |residual| / sqrt(context^T A^{-1} context)
        """
        try:
            L = self.L[arm]
            # A^{-1} context = L^{-T} L^{-1} context
            v = np.linalg.solve(L.T, np.linalg.solve(L, context))
            var = context @ v
            if var < 1e-12:
                return float('inf')
            return abs(residual) / np.sqrt(var)
        except Exception:
            return 0.0

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self._get_theta(a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Thompson Sampling: sample from posterior
            theta = self._get_theta(a)
            try:
                L = self.L[a]
                z = np.random.randn(self.d)
                noise = np.linalg.solve(L.T, z)
                theta_sample = theta + noise
                score = theta_sample @ context
            except Exception:
                score = theta @ context

            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction and residual
        theta = self._get_theta(arm)
        pred = theta @ context
        residual = reward - pred

        # Compute Mahalanobis distance to detect regime shift
        mahal_dist = self._mahalanobis_distance(arm, context, residual)

        # Adaptive forgetting: strong decay on regime shift, mild otherwise
        if mahal_dist > self.regime_threshold:
            decay = self.strong_decay
        else:
            decay = self.mild_decay

        # Update A and b with adaptive forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
        self.L[arm] = self._update_chol(self.A[arm])
# EVOLVE-BLOCK-END
