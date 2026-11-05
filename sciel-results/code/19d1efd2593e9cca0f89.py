"""
Multi-scale ensemble linear contextual bandit with adaptive model selection.
Each arm maintains a bank of regressors with geometrically spaced forgetting factors.
The policy selects the arm that maximizes the reward prediction from the ensemble member
with the lowest recent residual variance, effectively using model selection to dynamically
switch memory horizons.
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

        # Number of scales in the ensemble
        self.n_scales = 8
        # Forgetting factors: geometrically spaced from fast to slow
        # lambda_k = base * ratio^k, where base is the fastest (most forgetting)
        self.base_lambda = 0.90
        self.lambda_ratio = 1.05
        self.lambdas = [self.base_lambda * (self.lambda_ratio ** k) for k in range(self.n_scales)]
        # Clip to [0.8, 0.999]
        self.lambdas = [max(0.80, min(0.999, lam)) for lam in self.lambdas]

        # For each arm, for each scale, maintain A (covariance) and b (mean)
        # A[k][a] is a d x d matrix, b[k][a] is a d-vector
        self.A = [[np.eye(dim) * 1.0 for _ in range(n_arms)] for _ in range(self.n_scales)]
        self.b = [[np.zeros(dim) for _ in range(n_arms)] for _ in range(self.n_scales)]

        # Cholesky factors for each (scale, arm)
        self.L = [[np.linalg.cholesky(self.A[k][a]) for a in range(n_arms)] for k in range(self.n_scales)]

        # Residual tracking per (scale, arm) for model selection
        # We track the exponential moving average of squared residuals
        self.resid_ema = [[0.0 for _ in range(n_arms)] for _ in range(self.n_scales)]
        self.resid_ema_alpha = 0.1  # EMA decay for residual tracking

        # Initial confidence: all scales equal
        self.scale_weights = [1.0 / self.n_scales for _ in range(self.n_scales)]

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

    def _get_theta(self, scale, arm):
        """Get point estimate theta = A^{-1} b."""
        try:
            L = self.L[scale][arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[scale][arm]))
            return theta
        except Exception:
            return np.zeros(self.d)

    def _get_theta_cov(self, scale, arm):
        """Get theta and Cholesky factor for sampling."""
        try:
            L = self.L[scale][arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[scale][arm]))
            return theta, L
        except Exception:
            return np.zeros(self.d), np.eye(self.d)

    def _select_scale(self, arm):
        """Select the best scale for a given arm based on lowest residual EMA."""
        # Find the scale with the lowest residual EMA
        min_resid = float('inf')
        best_scale = 0
        for k in range(self.n_scales):
            if self.resid_ema[k][arm] < min_resid:
                min_resid = self.resid_ema[k][arm]
                best_scale = k
        return best_scale

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                k = self._select_scale(a)
                theta = self._get_theta(k, a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Use Thompson Sampling with the best scale for this arm
            best_scale = self._select_scale(a)
            theta, L = self._get_theta_cov(best_scale, a)

            try:
                # Sample from posterior
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

        # Update all scales for this arm
        for k in range(self.n_scales):
            lam = self.lambdas[k]

            # Compute prediction and residual
            theta = self._get_theta(k, arm)
            pred = theta @ context
            residual = reward - pred
            resid_sq = residual * residual

            # Update residual EMA
            self.resid_ema[k][arm] = (1 - self.resid_ema_alpha) * self.resid_ema[k][arm] + self.resid_ema_alpha * resid_sq

            # Update A and b with forgetting
            self.A[k][arm] = lam * self.A[k][arm] + np.outer(context, context)
            self.b[k][arm] = lam * self.b[k][arm] + reward * context
            self.L[k][arm] = self._update_chol(self.A[k][arm])
# EVOLVE-BLOCK-END
