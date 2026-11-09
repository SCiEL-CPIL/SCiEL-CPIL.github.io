"""Non-stationary linear contextual bandit: Thompson Sampling with recursive ridge regression and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with recursive ridge-regression posterior per arm.

    Uses Sherman-Morrison-Woodbury rank-1 updates for O(d^2) per step.
    Adaptive forgetting: when prediction residuals are high, we shrink the
    effective memory (increase forgetting) to track non-stationarity faster.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.ridge = 1.0
        self.noise = 1.0

        # Base forgetting factor (lambda): closer to 1 means longer memory
        self.base_lambda = 0.97
        # Adaptive range
        self.lambda_min = 0.90
        self.lambda_max = 0.995

        # Per-arm recursive ridge regression state
        self.A = [self.ridge * np.eye(dim) for _ in range(self.n)]
        self.b = [np.zeros(dim) for _ in range(self.n)]
        self.Ainv = [np.eye(dim) / self.ridge for _ in range(self.n)]
        self.theta = [np.zeros(dim) for _ in range(self.n)]

        # Per-arm adaptive forgetting
        self.lambdas = [self.base_lambda for _ in range(self.n)]

        # Per-arm residual tracking for adaptivity
        self.residual_ewma = [0.0 for _ in range(self.n)]
        self.residual_alpha = 0.3  # EWMA smoothing for residuals

        # Cholesky factors for sampling (recomputed periodically or on update)
        self.L = [np.eye(dim) for _ in range(self.n)]
        self._chol_dirty = [True for _ in range(self.n)]

        # Deadline guard
        self.t0 = time.time()
        self.deadline = 120.0
        self.safety_margin = 5.0

        self._rng = np.random.default_rng(42)

    def _update_arm(self, a, x, r):
        """Rank-1 Sherman-Morrison-Woodbury update for arm a."""
        d = self.d
        lam = self.lambdas[a]

        # Update A and b with forgetting
        # A_new = lam * A_old + x x^T
        # b_new = lam * b_old + r * x
        # Ainv_new = Ainv_old / lam + (Ainv_old @ x @ x^T @ Ainv_old) / (lam + x^T @ Ainv_old @ x)
        # Simplified: Ainv_new = Ainv_old + (1/lam - 1) * Ainv_old + correction for rank-1
        # More precisely:
        # A_new = lam * A_old + x x^T
        # Ainv_new = Ainv_old / lam - (Ainv_old @ x @ x^T @ Ainv_old) / (lam * (1/lam + x^T @ Ainv_old @ x))
        # Let me use the standard formula:
        # A_new = lam * A_old + x x^T
        # Ainv_new = (lam * A_old + x x^T)^{-1}
        # By SMW: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_old_scaled = lam * A_old, so A_old_scaled^{-1} = Ainv_old / lam
        # A_new = A_old_scaled + x x^T
        # Ainv_new = (Ainv_old / lam) - ((Ainv_old / lam) @ x @ x^T @ (Ainv_old / lam)) / (1 + x^T @ (Ainv_old / lam) @ x)

        Ainv = self.Ainv[a]
        Ainv_over_lam = Ainv / lam

        # Compute scalar denominator
        Ax = Ainv_over_lam @ x
        denom = 1.0 + float(x @ Ax)

        # SMW update
        if abs(denom) > 1e-12:
            Ainv_new = Ainv_over_lam - np.outer(Ax, Ax) / denom
        else:
            # Fallback: recompute from A
            self.A[a] = lam * self.A[a] + np.outer(x, x)
            Ainv_new = np.linalg.inv(self.A[a])

        self.Ainv[a] = Ainv_new

        # Update b
        self.b[a] = lam * self.b[a] + r * x

        # Update theta = Ainv @ b
        self.theta[a] = Ainv_new @ self.b[a]

        # Update residual EWMA for adaptivity
        pred = float(self.theta[a] @ x)
        residual = abs(r - pred)
        self.residual_ewma[a] = (1 - self.residual_alpha) * self.residual_ewma[a] + self.residual_alpha * residual

        # Adapt forgetting factor based on residual
        # High residual -> smaller lambda (faster forgetting)
        # Low residual -> larger lambda (longer memory)
        if self.residual_ewma[a] > 2.0:
            self.lambdas[a] = max(self.lambda_min, self.lambdas[a] * 0.98)
        elif self.residual_ewma[a] < 0.5:
            self.lambdas[a] = min(self.lambda_max, self.lambdas[a] * 1.02)
        else:
            # Slowly drift back to base
            self.lambdas[a] += 0.01 * (self.base_lambda - self.lambdas[a])

        # Mark Cholesky as dirty
        self._chol_dirty[a] = True

    def _get_theta_and_L(self, a):
        """Return theta and Cholesky factor for covariance of arm a."""
        if self._chol_dirty[a]:
            # Covariance = noise * Ainv
            cov = self.noise * self.Ainv[a]
            # Ensure symmetry
            cov = (cov + cov.T) / 2.0
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # Add small jitter
                cov += 1e-6 * np.eye(self.d)
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    L = np.eye(self.d) * np.sqrt(self.noise)
            self.L[a] = L
            self._chol_dirty[a] = False
        return self.theta[a], self.L[a]

    def select(self, context):
        context = np.asarray(context, dtype=float).ravel()

        # Deadline guard
        if (time.time() - self.t0) > (self.deadline - self.safety_margin):
            # Greedy fallback
            scores = [float(self.theta[a] @ context) for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta, L = self._get_theta_and_L(a)
            # Thompson sampling: sample from N(theta, L L^T)
            z = self._rng.standard_normal(self.d)
            theta_sample = theta + L @ z
            scores.append(float(theta_sample @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float).ravel()
        self._update_arm(arm, context, float(reward))
# EVOLVE-BLOCK-END
