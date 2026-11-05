"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with adaptive forgetting via EMA of Mahalanobis residuals."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive decay parameters
        self.base_decay = 0.98
        self.min_decay = 0.85
        self.max_decay = 0.999
        # EMA of squared Mahalanobis distance for smooth adaptation
        self.ema_mahal_sq = 0.0
        self.ema_alpha = 0.15
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        # Precompute Cholesky factors
        self._chols = [np.eye(dim) for _ in range(n_arms)]

    def _adaptive_decay(self, arm, context, reward):
        """Compute decay factor from EMA of Mahalanobis residual."""
        try:
            L = self._chols[arm]
            # theta = A^{-1} b
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            pred = theta @ context
            residual = reward - pred
            # context^T A^{-1} context
            A_inv_x = np.linalg.solve(L.T, np.linalg.solve(L, context))
            ctx_var = context @ A_inv_x
            mahal_sq = (residual ** 2) / (ctx_var + 1e-8)
            # Update EMA
            self.ema_mahal_sq = (1 - self.ema_alpha) * self.ema_mahal_sq + self.ema_alpha * mahal_sq
            # Map EMA to decay: higher EMA -> lower decay (faster forgetting)
            # Normalize: if EMA ~ 1 (noise level), use base_decay
            # If EMA >> 1, reduce decay significantly
            ratio = self.ema_mahal_sq / 1.0  # 1.0 is expected under stationarity
            if ratio > 1.0:
                # Exponentially reduce decay as ratio increases
                target = self.base_decay * np.exp(-0.3 * (ratio - 1.0))
                target = max(target, self.min_decay)
            else:
                target = self.base_decay
            return target
        except Exception:
            return self.base_decay

    def select(self, context):
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    L = self._chols[a]
                    theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except Exception:
                    pass
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for a in range(self.n):
            try:
                L = self._chols[a]
                theta_hat = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                z = np.random.randn(self.d)
                # Sample: theta = theta_hat + L^{-T} z
                theta_sample = theta_hat + np.linalg.solve(L.T, z)
                scores[a] = theta_sample @ context
            except Exception:
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores[a] = theta @ context
                except Exception:
                    scores[a] = 0.0
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute adaptive decay
        decay = self._adaptive_decay(arm, context, reward)

        # Update A and b with forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context

        # Update Cholesky factorization
        try:
            self._chols[arm] = np.linalg.cholesky(self.A[arm] + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self.A[arm] = self.A[arm] + 1e-6 * np.eye(self.d)
            try:
                self._chols[arm] = np.linalg.cholesky(self.A[arm])
            except Exception:
                self._chols[arm] = np.eye(self.d)
# EVOLVE-BLOCK-END
