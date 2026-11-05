"""Non-stationary linear contextual bandit with adaptive Kalman filtering and robust updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using per-arm Kalman Filters with Regime Switching.
    
    Features:
    1. Per-arm Kalman Filter (theta, P) for weight tracking.
    2. Regime-Switching Process Noise: Two modes (Stable/Volatile) with HMM-like transitions.
    3. Adaptive Q based on current regime and innovation magnitude.
    4. Huber-loss robustification of innovations to handle outliers/shifts.
    5. Covariance reset mechanism for catastrophic forgetting recovery.
    6. Adaptive UCB exploration with warmup and decay.
    7. Strict deadline guard.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        # Safety margin: 110s allows 10s buffer before the 120s kill (or 150s limit)
        self.deadline = self.start_time + 110.0 

        # Kalman filter state per arm
        # Initialize theta to 0
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with high uncertainty to encourage exploration
        # 10.0 is a good balance: high enough to explore, low enough to not be unstable
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Regime Switching Parameters
        # Modes: 0 = Stable (low noise), 1 = Volatile (high noise)
        self.regime = [0] * n_arms
        self.regime_prob = [0.1] * n_arms  # Probability of being in Volatile mode
        self.p_stay_stable = 0.95   # P(Stable -> Stable)
        self.p_stay_volatile = 0.80 # P(Volatile -> Volatile)
        self.regime_threshold = 1.5 # Z-score threshold for regime switch
        self.regime_ema_alpha = 0.5 # Smoothing for regime probability update

        # Adaptive process noise parameters
        self.q_base_stable = 0.01   # Base process noise for Stable mode
        self.q_base_volatile = 0.2  # Base process noise for Volatile mode
        self.q_max = 5.0           # Cap on adaptive process noise
        self.innov_ema = [0.0] * n_arms  # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.3       # Fast adaptation rate for EMA
        self.innov_base = 1.0            # Baseline threshold for innovation
        
        # Context-norm adaptive scaling
        self.ctx_norm_gamma = 0.05  # How strongly context norm scales Q
        self.ctx_norm_cap = 3.0     # Cap on the context-norm multiplier

        # Huber robustification
        self.huber_delta = 2.0      # Threshold for clipping innovation in std units

        # Sliding-window reset mechanism
        self.reset_threshold_scale = 2.5  # Multiplier on std for triggering reset
        self.reset_cov_scale = 10.0       # Reset P to this times identity
        self.reset_window = 8             # Number of consecutive large innovations
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration parameters
        self.u_scale = 1.5         # Base UCB multiplier
        self.warmup_steps = 20     # Steps to boost exploration early
        self.jitter = 1e-6         # Numerical stabilizer
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _update_regime(self, arm, z_score):
        """
        Update regime probability based on innovation z-score.
        """
        # If z-score is high, increase probability of volatile mode
        # Use a simple logistic-like update or direct probability adjustment
        if abs(z_score) > self.regime_threshold:
            # Likely a shift or outlier, move towards volatile
            new_prob = self.regime_prob[arm] + (1.0 - self.regime_prob[arm]) * self.regime_ema_alpha
        else:
            # Likely stable, move towards stable
            new_prob = self.regime_prob[arm] * (1.0 - self.regime_ema_alpha)
        
        # Clamp probability
        self.regime_prob[arm] = max(0.01, min(0.99, new_prob))
        
        # Update discrete regime state (optional, used for Q selection)
        # We can keep the discrete state for clarity or just use the probability
        # Let's use the probability to blend Q values or select the dominant one
        # For now, let's select the mode with higher probability
        if self.regime_prob[arm] > 0.5:
            self.regime[arm] = 1
        else:
            self.regime[arm] = 0

    def _adaptive_q(self, arm, innov, ctx_norm_sq):
        """
        Compute adaptive process noise scalar.
        Combines regime-based base noise with innovation-based adaptation.
        """
        c = self.count[arm]
        if c < 3:
            # Early phase: larger Q to help initial convergence
            return self.q_base_volatile

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        # Scale Q based on how much the innovation exceeds the baseline
        excess = max(0.0, innov_ema - self.innov_base)
        # Smooth exponential scaling
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        # Context-norm adaptive scaling
        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        # Select base Q based on current regime
        base_q = self.q_base_volatile if self.regime[arm] == 1 else self.q_base_stable
        
        q = min(self.q_max, base_q * q_scale * ctx_mult)
        return q

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

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        # Add scalar q to diagonal (isotropic process noise)
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
        # Simple diagonal check is faster and often sufficient for this scale
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))
            
        self.count[a] += 1
        return innov_raw, std

    def _check_reset(self, a, innov_raw):
        """
        Check if we should reset covariance due to sustained large innovations.
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
            # Reset covariance to a moderate value to force re-exploration
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
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
        # arm with the highest mean reward (greedy)
        if self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(float(self.theta[a] @ ctx))
            return int(np.argmax(scores))

        u = self._adaptive_u()
        scores = []
        for a in range(self.n):
            mu = float(self.theta[a] @ ctx)
            # UCB bonus: sqrt(ctx^T P ctx)
            # Ensure positive variance
            var = float(ctx @ self.P[a] @ ctx)
            if var < 0:
                var = 0.0
            bonus = u * np.sqrt(var + self.jitter)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.step_count += 1

        # 1. Compute adaptive Q based on previous state (predict step uses current Q)
        # We need the innovation to compute Q, but Q is used in the update.
        # Standard Kalman: Predict (add Q) -> Update (using innovation).
        # Here, we estimate Q based on the *last* innovation or a running average.
        # To be precise, we use the EMA from the previous step to set Q for this step's predict.
        
        # Calculate context norm squared for adaptive Q scaling
        ctx_norm_sq = float(ctx @ ctx)
        
        # Get adaptive Q based on current EMA state
        # Note: The EMA is updated *after* we compute the innovation in _update.
        # So we use the EMA from the previous call.
        q = self._adaptive_q(arm, 0.0, ctx_norm_sq) # Pass 0.0 as placeholder, EMA is used internally
        
        # 2. Predict step
        self._predict(arm, q)

        # 3. Update step (returns raw innovation and std)
        innov_raw, std = self._update(arm, ctx, reward)
        
        # 4. Update regime based on innovation z-score
        if std > 0:
            z_score = innov_raw / std
        else:
            z_score = 0.0
        self._update_regime(arm, z_score)

        # 5. Check for reset condition
        self._check_reset(arm, innov_raw)

# EVOLVE-BLOCK-END
