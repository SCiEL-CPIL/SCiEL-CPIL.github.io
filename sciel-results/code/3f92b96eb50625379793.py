"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    - Huber robustification for outlier handling
    - Context-norm adaptive exploration scaling
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # safety margin before 120s evaluator limit

        # Inverse covariance matrices V = A^{-1} for each arm
        ridge = 0.15
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters — tuned forgetting for drift tracking
        self.base_gamma = 0.93
        self.gamma_min = 0.72
        self.gamma_max = 0.985
        self.C = 1.35
        self.C_min = 0.25
        self.C_max = 4.5

        # For tracking prediction errors to adapt forgetting
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.35

        # Huber robustification parameters
        self.huber_delta = 2.0

        # Context-norm adaptive exploration scaling
        self.ctx_norm_gamma = 0.05
        self.ctx_norm_cap = 3.0

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction error (EMA)."""
        c = self.counts[arm]
        if c < 3:
            return self.base_gamma

        err_ema = self.pred_err_ema[arm]
        err_scale = min(err_ema, 5.0)
        # Smooth, monotonic mapping with stronger response
        gamma = self.base_gamma - 0.15 * (1.0 - np.exp(-0.8 * err_scale))
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm, ctx_norm_sq):
        """Adapt exploration based on count, time, and context norm."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 6:
            return self.C * 1.4
        # Smooth exponential decay of exploration over time
        time_decay = np.exp(-self.t / 1000.0)
        time_decay = max(0.35, time_decay)
        # Context-norm adaptive scaling
        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)
        return self.C * time_decay * ctx_mult

    def _huber_innov(self, innov, std):
        """
        Robustify innovation using Huber loss logic.
        If the innovation is larger than delta * std, clip it to delta * std.
        """
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        # For |innov|/std > delta, use linear scaling (clipping)
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        ctx_norm_sq = float(context @ context)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus with context-norm adaptive scaling
            c_val = self._adaptive_C(a, ctx_norm_sq)
            bonus = c_val * np.sqrt(max(0.0, context @ self.V[a] @ context))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        
        # Innovation variance estimate for Huber robustification
        # Using the current inverse covariance to estimate uncertainty
        var = float(context @ self.V[arm] @ context)
        std = float(np.sqrt(max(var, 0.0)))
        
        # Robustify the error for the EMA update to avoid outlier domination
        err_robust = self._huber_innov(err, std)
        
        # Update EMA of squared robust error
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + \
                                  self.err_ema_alpha * (err_robust * err_robust)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-10:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
            # Ensure symmetry
            self.V[arm] = 0.5 * (self.V[arm] + self.V[arm].T)
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d) * (1.0 + 0.1)

        # Update b with forgetting, using robustified reward signal
        # Note: We use the raw reward for the linear update to maintain unbiasedness in expectation,
        # but the adaptive gamma is driven by robust error.
        # However, if we want to be fully robust against outliers in the weight estimate itself,
        # we might consider clipping the reward. Let's stick to robust gamma for now,
        # but we can also robustify the reward update if the error is huge.
        # For now, let's just use the raw reward in the linear combination,
        # as the forgetting mechanism handles the "forgetting" of bad data.
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic numerical stabilization: every 100 steps, add small ridge
        if self.t % 100 == 0:
            for a in range(self.n):
                self.V[a] += 0.003 * np.eye(self.d)
# EVOLVE-BLOCK-END
