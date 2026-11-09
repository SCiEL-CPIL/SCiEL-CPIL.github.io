import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Core design (kept from parent family):
    - Per-arm discounted linear regression (discounted LinUCB).
    - UCB-style action selection: mean + beta * std.
    - Global non-stationarity awareness via a smoothed error signal that modulates beta.
    
    New refinements (within the same family):
    1) Adaptive, smoothed discount factor:
       - We track per-arm error variance (EMA of squared prediction error).
       - Arms with high recent error get a smaller alpha (faster forgetting) to adapt quicker,
       - Arms with low error keep a larger alpha (slower forgetting) for stability.
       This is a smooth, monotone mapping from error to alpha.
    2) Smoother, more principled beta modulation:
       - Use a log-sum-exp-like soft-max on normalized per-arm prediction errors to produce
         a global "instability" score in [0,1]. This is a smooth surrogate for a max/min
         objective and avoids hard thresholds.
       - beta = base_beta + (max_beta - base_beta) * instability.
    3) Numerical stabilization:
       - Use Cholesky-based solves when possible for theta and A^{-1}x, falling back to
         lstsq/inv.
       - Clip small negative variances to zero.
    4) Deadline guard with margin.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Regularization
        self.reg = 1.0

        # Per-arm model state
        self.A = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Base and adaptive discount parameters
        self.alpha_base = 0.99
        self.alpha_min = 0.92   # faster forgetting
        self.alpha_max = 0.999  # slower forgetting

        # Error tracking for adaptive alpha
        # EMA of squared prediction error per arm
        self.err2_ema = [0.0 for _ in range(n_arms)]
        self.err2_alpha = 0.2  # EMA rate for squared error

        # Global instability tracking (smoothed)
        self.instability = 0.0
        self.instab_alpha = 0.15

        # UCB exploration parameters
        self.base_beta = 1.4
        self.max_beta = 3.0

        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def _adaptive_alpha(self, arm):
        """
        Map recent squared-error EMA to a discount factor.
        High error -> smaller alpha (faster forgetting).
        Low error  -> larger alpha (slower forgetting).
        We use a simple smooth, monotone mapping:
            alpha = alpha_max - (alpha_max - alpha_min) * s
        where s in [0,1] is a normalized error signal.
        """
        # Normalize error using a soft scale to avoid hard thresholds
        # We use a log(1 + err2) to compress large values
        e = np.log1p(self.err2_ema[arm])
        # Map to [0,1] via a smooth saturating function (tanh)
        s = 0.5 * (1.0 + np.tanh(0.5 * e))
        alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * s
        # Clamp to [alpha_min, alpha_max]
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        return alpha

    def _soft_instability(self, context):
        """
        Compute a smooth global instability score in [0,1].
        We use a log-sum-exp (soft-max) over per-arm normalized prediction errors
        to emphasize the worst arm without a hard max.
        """
        if self.d <= 0:
            return 0.0

        context = np.asarray(context, dtype=float)

        # Compute per-arm prediction errors (absolute)
        errs = np.zeros(self.n)
        for a in range(self.n):
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                pred = theta @ context
                # We don't have the reward here, so use a proxy:
                # Use the residual from the last update? Not available in select.
                # Instead, use a proxy based on how "stale" the arm is:
                # We'll use the magnitude of the current prediction relative to a baseline,
                # combined with the per-arm error EMA.
                # A simpler and more stable proxy: use the per-arm err2_ema directly.
                errs[a] = np.sqrt(self.err2_ema[a])
            except Exception:
                errs[a] = 1.0

        # Normalize errors to [0,1] using a soft scale
        # Use a log-sum-exp style: soft_max of scaled errors
        scale = 2.0
        scaled = scale * errs
        # Subtract max for numerical stability
        if scaled.size > 0:
            m = np.max(scaled)
            if np.isfinite(m):
                scaled_shifted = scaled - m
                # log-sum-exp
                lse = m + np.log(np.sum(np.exp(scaled_shifted)))
                # Soft-max weights
                w = np.exp(scaled_shifted) / np.sum(np.exp(scaled_shifted))
                # Instability as weighted average of normalized errors
                # Normalize each error to [0,1] via tanh
                norm_errs = 0.5 * (1.0 + np.tanh(errs))
                instab = float(np.dot(w, norm_errs))
            else:
                instab = 0.0
        else:
            instab = 0.0

        # Smooth over time
        self.instability = (1 - self.instab_alpha) * self.instability + self.instab_alpha * instab
        return self.instability

    def _get_theta_and_cov(self, A, b):
        """
        Compute theta = A^{-1} b and the vector A^{-1} x (for a given x, if needed).
        Returns (theta, A_inv) with fallbacks.
        """
        try:
            # Try Cholesky for speed and stability
            L = np.linalg.cholesky(A)
            # Solve L y = b, then L^T theta = y
            y = np.linalg.solve(L, b)
            theta = np.linalg.solve(L.T, y)
            # Compute A_inv via Cholesky
            # A^{-1} = L^{-T} L^{-1}
            I = np.eye(self.d)
            Linv = np.linalg.inv(L)
            A_inv = Linv.T @ Linv
            return theta, A_inv
        except np.linalg.LinAlgError:
            # Fallback to general solve and inverse
            try:
                theta = np.linalg.solve(A, b)
                A_inv = np.linalg.inv(A)
                return theta, A_inv
            except np.linalg.LinAlgError:
                # Last resort: least squares
                try:
                    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
                    A_inv = np.linalg.pinv(A)
                    return theta, A_inv
                except Exception:
                    return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0  # Fallback to first arm to avoid timeout

        context = np.asarray(context, dtype=float)

        best_arm = 0
        best_score = -np.inf

        # Compute global instability (smoothed)
        instab = self._soft_instability(context)
        current_beta = self.base_beta + (self.max_beta - self.base_beta) * instab

        for a in range(self.n):
            try:
                theta, A_inv = self._get_theta_and_cov(self.A[a], self.b[a])

                # Expected reward
                mean_reward = theta @ context

                # Uncertainty: x^T A^{-1} x
                variance = context @ A_inv @ context
                if variance < 0:
                    variance = 0.0
                std_dev = np.sqrt(variance)

                # UCB Score
                score = mean_reward + current_beta * std_dev

                if score > best_score:
                    best_score = score
                    best_arm = a

            except Exception:
                # Fallback: use mean only
                try:
                    theta, _ = self._get_theta_and_cov(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except Exception:
                    continue

        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)

        # 1. Compute prediction error for the selected arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_sq = error * error
        except Exception:
            error_sq = 1.0

        # 2. Update per-arm squared-error EMA
        self.err2_ema[arm] = (1 - self.err2_alpha) * self.err2_ema[arm] + self.err2_alpha * error_sq

        # 3. Adaptive discount factor for the selected arm
        alpha_sel = self._adaptive_alpha(arm)

        # 4. Update model for the selected arm with discounting
        self.A[arm] = alpha_sel * self.A[arm] + np.outer(context, context)
        self.b[arm] = alpha_sel * self.b[arm] + reward * context

        # 5. Discount other arms (use their own adaptive alpha)
        for a in range(self.n):
            if a != arm:
                alpha_other = self._adaptive_alpha(a)
                self.A[a] = alpha_other * self.A[a]
                self.b[a] = alpha_other * self.b[a]

        # 6. Time guard (no-op, but keeps structure)
        if time.time() > self.deadline:
            pass
