# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a per-arm Robust Kalman Filter
    with Huber-weighted innovations, adaptive process noise, adaptive observation
    noise, and UCB exploration.

    Key features:
    1. Per-arm Kalman Filter for weight tracking.
    2. Adaptive Process Noise (Q) based on innovation EMA and context norm.
    3. Smooth Huber-weighted innovations to robustly down-weight outliers.
    4. Adaptive observation noise (R) tracking per-arm reward variance.
    5. Sliding-window covariance reset for catastrophic forgetting recovery.
    6. Adaptive UCB exploration with warmup and decay.
    7. Strict deadline guard to ensure completion within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

        # Kalman filter state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive process noise parameters
        self.q_base = 0.05
        self.q_max = 5.0
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.3
        self.innov_base = 1.0

        # Context-norm adaptive scaling
        self.ctx_norm_gamma = 0.05
        self.ctx_norm_cap = 3.0

        # Huber weighting parameters
        self.huber_delta = 2.0

        # Adaptive observation noise (R) per arm
        self.R = [1.0] * n_arms
        self.R_min = 0.1
        self.R_max = 10.0
        self.R_ema_alpha = 0.2
        self.R_innov_ema = [1.0] * n_arms  # EMA of |innov| per arm for R adaptation

        # Sliding-window reset mechanism
        self.reset_threshold_scale = 2.5
        self.reset_cov_scale = 10.0
        self.reset_window = 8
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration parameters
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov, ctx_norm_sq):
        """
        Compute adaptive process noise scalar.
        Combines innovation-based adaptation with context-norm scaling.
        """
        c = self.count[arm]
        if c < 3:
            return self.q_base * 2.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        q = min(self.q_max, self.q_base * q_scale * ctx_mult)
        return q

    def _huber_weight(self, innov, std):
        """
        Compute a smooth Huber-based weight for the innovation.
        For |innov|/std <= delta: weight = 1.0 (full trust)
        For |innov|/std > delta: weight = delta * std / |innov| (linear down-weight)
        This is smooth at the boundary and avoids hard clipping discontinuity.
        """
        if std <= 0:
            return 1.0
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return 1.0
        return self.huber_delta / ratio

    def _adaptive_R(self, arm, innov_raw, std):
        """
        Adapt observation noise R based on the magnitude of the innovation.
        Large innovations suggest either a regime shift (handled by Q) or
        higher-than-expected observation noise. We track an EMA of |innov|
        and use it to modulate R.
        """
        abs_innov = abs(innov_raw)
        # Update EMA of absolute innovation
        self.R_innov_ema[arm] = (1.0 - self.R_ema_alpha) * self.R_innov_ema[arm] + \
                                 self.R_ema_alpha * abs_innov

        # Scale R: if innovations are consistently large, increase R to
        # reduce the Kalman gain (trust the model less per observation)
        ema_val = self.R_innov_ema[arm]
        # Normalize: if ema_val > 1, increase R proportionally but capped
        r_scale = 1.0 + 0.5 * max(0.0, ema_val - 1.0)
        r_scale = min(r_scale, self.R_max / self.R_min)

        new_R = min(self.R_max, max(self.R_min, 1.0 * r_scale))
        self.R[arm] = new_R
        return new_R

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """
        Robust Kalman update step with Huber-weighted innovation.
        Returns the raw innovation for tracking purposes.
        """
        P = self.P[a]
        theta = self.theta[a]

        pred = float(theta @ ctx)
        innov_raw = reward - pred

        # Innovation variance: S = x^T P x + R
        R = self.R[a]
        var = float(ctx @ P @ ctx) + R + self.jitter
        std = float(np.sqrt(var))

        # Smooth Huber weight: down-weight large innovations
        w = self._huber_weight(innov_raw, std)

        # Effective innovation with Huber weighting
        innov_weighted = w * innov_raw

        # Kalman gain: K = P x / S
        K = (P @ ctx) / var

        # Update mean: theta <- theta + K * innov_weighted
        self.theta[a] = theta + (K * innov_weighted)

        # Joseph-form covariance update for numerical stability:
        # P <- (I - K x^T) P (I - K x^T)^T
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T

        # Ensure symmetry
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)

        # Clip diagonal for stability
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))

        self.count[a] += 1
        return innov_raw

    def _check_reset(self, a, innov_raw):
        """
        Check if we should reset covariance due to sustained large innovations.
        """
        P = self.P[a]
        var = float(np.trace(P)) / self.d
        std = float(np.sqrt(max(var, 0.0)))
        threshold = self.reset_threshold_scale * std + 0.5

        if abs(innov_raw) > threshold:
            self.consecutive_large_innov[a] += 1
        else:
            self.consecutive_large_innov[a] = 0

        if self.consecutive_large_innov[a] >= self.reset_window:
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
            # Also reset R to default to avoid stuck high-R state
            self.R[a] = 1.0
            self.R_innov_ema[a] = 1.0
            return True
        return False

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        if self._check_deadline():
            scores = [float(self.theta[a] @ ctx) for a in range(self.n)]
            return int(np.argmax(scores))

        u = self._adaptive_u()
        scores = []

        for a in range(self.n):
            pred = float(self.theta[a] @ ctx)
            var = float(ctx @ self.P[a] @ ctx)
            std = float(np.sqrt(max(var, 0.0)))
            score = pred + u * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.step_count += 1

        ctx_norm_sq = float(ctx @ ctx)

        # Compute raw innovation for adaptive Q and R
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred

        # Compute adaptive Q using this innovation
        q = self._adaptive_q(arm, innov_raw, ctx_norm_sq)

        # Compute adaptive R using this innovation
        # First get std estimate from current P
        P = self.P[arm]
        var_est = float(ctx @ P @ ctx) + self.R[arm]
        std_est = float(np.sqrt(max(var_est, 1e-8)))
        R_new = self._adaptive_R(arm, innov_raw, std_est)

        # Apply predict step
        self._predict(arm, q)

        # Apply update step (uses updated R)
        innov_from_update = self._update(arm, ctx, reward)

        # Check for reset based on the raw innovation
        self._check_reset(arm, innov_raw)
# EVOLVE-BLOCK-END
