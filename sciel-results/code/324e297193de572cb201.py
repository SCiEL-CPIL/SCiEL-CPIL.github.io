# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a per-arm Kalman filter
    with adaptive process noise, Huber robustification, and UCB exploration.
    
    Key features:
    1. Per-arm Kalman Filter for weight tracking.
    2. Adaptive Process Noise (Q) based on innovation EMA and context norm.
    3. Huber-loss based innovation clipping to handle outliers/shifts robustly.
    4. Sliding-window covariance reset for catastrophic forgetting recovery.
    5. Adaptive UCB exploration with warmup and decay.
    6. Strict deadline guard to ensure completion within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        # Safety margin: 110s allows 10s buffer before the 120s kill
        self.deadline = self.start_time + 110.0 

        # Kalman filter state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize with high uncertainty to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive process noise parameters
        self.q_base = 0.05         # Base process noise
        self.q_max = 5.0           # Cap on adaptive process noise
        self.innov_ema = [0.0] * n_arms  # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.3       # Fast adaptation rate for EMA
        self.innov_base = 1.0            # Baseline threshold for innovation
        
        # Context-norm adaptive scaling (from Inspiration 1)
        self.ctx_norm_gamma = 0.05  # How strongly context norm scales Q
        self.ctx_norm_cap = 3.0     # Cap on the context-norm multiplier

        # Huber robustification (from Inspiration 1)
        self.huber_delta = 2.0      # Threshold for clipping innovation in std units

        # Sliding-window reset mechanism (from Inspiration 1)
        self.reset_threshold_scale = 2.5  # Multiplier on std for triggering reset
        self.reset_cov_scale = 10.0       # Reset P to this times identity
        self.reset_window = 8             # Number of consecutive large innovations
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration parameters
        self.u_scale = 1.5         # Base UCB multiplier
        self.count_scale = 0.4     # Dampening factor for bonus based on count
        self.warmup_steps = 20     # Steps to boost exploration early
        self.jitter = 1e-6         # Numerical stabilizer
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
            # Early phase: larger Q to help initial convergence
            return self.q_base * 2.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        # Scale Q based on how much the innovation exceeds the baseline
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        # Context-norm adaptive scaling
        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        q = min(self.q_max, self.q_base * q_scale * ctx_mult)
        return q

    def _huber_innov(self, innov, std):
        """
        Robustify innovation using Huber loss logic.
        If the innovation is larger than delta * std, clip it to delta * std.
        This prevents single large outliers from dominating the update.
        """
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        # For |innov|/std > delta, use linear scaling (clipping)
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """
        Kalman update step: rank-1 update with robustified innovation.
        Returns the raw innovation for tracking purposes.
        """
        P = self.P[a]
        theta = self.theta[a]
        
        # Predicted reward
        pred = float(theta @ ctx)
        # Raw innovation
        innov_raw = reward - pred
        
        # Innovation variance: S = x^T P x + R (assuming R=1)
        R = 1.0
        var = float(ctx @ P @ ctx) + R + self.jitter
        std = float(np.sqrt(var))
        
        # Robustify innovation
        innov_robust = self._huber_innov(innov_raw, std)
        
        # Kalman gain: K = P x / S
        K = (P @ ctx) / var
        
        # Update mean: theta <- theta + K * innov_robust
        self.theta[a] = theta + (K * innov_robust)
        
        # Joseph-form covariance update for numerical stability:
        # P <- (I - K x^T) P (I - K x^T)^T
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T
        
        # Ensure symmetry and positive semi-definite
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        
        # Clip eigenvalues to ensure stability
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))
            
        self.count[a] += 1
        return innov_raw

    def _check_reset(self, a, innov_raw):
        """
        Check if we should reset covariance due to sustained large innovations.
        This handles cases where the filter might be tracking a shifted regime
        poorly and needs a 'hard' restart of uncertainty.
        """
        P = self.P[a]
        # Use average variance as a proxy for overall uncertainty
        var = float(np.trace(P)) / self.d 
        std = float(np.sqrt(max(var, 0.0)))
        threshold = self.reset_threshold_scale * std + 0.5

        if abs(innov_raw) > threshold:
            self.consecutive_large_innov[a] += 1
        else:
            self.consecutive_large_innov[a] = 0

        if self.consecutive_large_innov[a] >= self.reset_window:
            # Reset covariance to a moderate value
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
            # Optionally, we could also dampen theta, but resetting P is the primary fix
            return True
        return False

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            # Decay exploration slightly over time, but keep a floor
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest predicted reward (greedy)
        if self._check_deadline():
            scores = [float(self.theta[a] @ ctx) for a in range(self.n)]
            return int(np.argmax(scores))

        u = self._adaptive_u()
        scores = []
        
        # Precompute ctx_norm_sq for potential use in Q if we were predicting here,
        # but Q is only updated in the update step. For selection, we just use current P.
        
        for a in range(self.n):
            # Prediction
            pred = float(self.theta[a] @ ctx)
            
            # Uncertainty
            # P is already updated with process noise in the previous step's update,
            # or initialized. We don't need to add Q again here unless we strictly
            # separate predict/update phases. The current design adds Q in _predict
            # which is called in update. So P in select() is the "prior" for the
            # current context if we consider the last update as having completed
            # the predict+update cycle for the previous step.
            
            # Wait, the standard Kalman loop is:
            # 1. Select (using current theta, P)
            # 2. Update (with new reward)
            #    - Predict: P += Q
            #    - Update: theta, P update
            
            # So in select(), P is the posterior from the last update.
            # It does NOT include the process noise for the *current* step yet.
            # This is correct because the process noise represents the change
            # that happens *after* the last observation and *before* the next.
            # So when we select, we should ideally use P_prior = P_posterior + Q.
            # However, Q depends on the innovation from the *last* step, which we have.
            # But the context for the *current* step is different.
            # For simplicity and to match the high-performing inspiration, we use
            # the current P as the uncertainty estimate. This is a common approximation.
            
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

        # Calculate context norm squared for adaptive Q
        ctx_norm_sq = float(ctx @ ctx)

        # Predict step: add adaptive process noise
        # We need the innovation to compute adaptive Q, but innovation requires
        # prediction. We use the raw innovation from the *previous* step?
        # No, the adaptive Q in the inspiration uses the innovation *of the current step*
        # to update the EMA, but the Q is applied *before* the update.
        # Let's look at the inspiration logic:
        # _adaptive_q(arm, innov, ctx_norm_sq)
        # It updates the EMA inside _adaptive_q.
        # So we need to compute the raw innovation first?
        # Raw innovation = reward - theta @ ctx.
        # This is the innovation *before* the update.
        
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred
        
        # Compute adaptive Q using this innovation
        q = self._adaptive_q(arm, innov_raw, ctx_norm_sq)
        
        # Apply predict step
        self._predict(arm, q)
        
        # Apply update step
        # _update returns the raw innovation again, but we already have it.
        # We call _update to perform the Kalman gain calculation and state update.
        # Note: _update recalculates pred and innov_raw, which is slightly redundant
        # but ensures consistency with the internal state.
        innov_from_update = self._update(arm, ctx, reward)
        
        # Check for reset based on the raw innovation
        self._check_reset(arm, innov_raw)
# EVOLVE-BLOCK-END
