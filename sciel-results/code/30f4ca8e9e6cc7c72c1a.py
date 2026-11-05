"""Non-stationary linear contextual bandit with dual-timescale Kalman filter and regime-switching."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a dual-timescale Kalman filter
    with regime-switching process noise.
    
    Core Idea:
    - Model arm weights as a random walk (process noise Q).
    - Use Kalman Filter updates for state estimation.
    - Dual-timescale: 
        1. Fast scale: Standard KF update for reward prediction.
        2. Slow scale: Adaptive Process Noise (Q) based on prediction error.
    - Regime Switching:
        - Maintain two modes: 'stable' (low Q) and 'volatile' (high Q).
        - Use a lightweight HMM-like switching based on innovation magnitude.
        - This allows explicit distinction between gradual drift and abrupt breaks.
    
    State:
    - theta[a]: Estimated weight vector for arm a.
    - P[a]: Covariance matrix for theta[a].
    - mode[a]: 0 for stable, 1 for volatile.
    
    Update:
    - Prediction: theta_pred = theta, P_pred = P + Q (Random Walk)
    - Update: Standard KF update with observation y = reward, H = context.
    - Regime Update: Adjust mode based on innovation size and EMA of errors.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 135.0  # Safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with moderate value to allow rapid initial learning
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Process noise base and adaptation params
        # Two modes: stable (low Q) and volatile (high Q)
        self.q_stable = 0.003
        self.q_volatile = 0.04
        self.q_min = 0.001
        self.q_max = 0.1
        
        # Regime switching parameters
        self.mode = [0] * n_arms  # 0: stable, 1: volatile
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # EMA smoothing factor
        self._switch_threshold_stable = 1.5  # Switch to volatile if innov_sq > this
        self._switch_threshold_volatile = 0.5  # Switch to stable if innov_sq < this
        
        # Adaptation params for Q
        self.error_threshold = 1.0  # Scale of expected noise
        self.adapt_gain = 0.5       # How aggressively to adapt Q
        
        # Exploration
        # UCB style: mu + alpha * sqrt(context^T P context)
        self.alpha = 1.2
        self.alpha_decay = 0.999
        self.alpha_min = 0.3
        
        # Tracking
        self.round_num = 0
        self.arm_counts = [0] * n_arms
        self.pred_errors = [np.zeros(10) for _ in range(n_arms)] # Circular buffer
        self.pred_error_idx = [0] * n_arms
        
        # Reset mechanism for catastrophic drift
        self._consec_large = [0] * n_arms
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adapt_q(self, arm, pred_error):
        """
        Adapt process noise Q based on recent prediction errors and current mode.
        High error -> High Q (expecting change).
        Low error -> Low Q (stable).
        """
        c = self.arm_counts[arm]
        if c < 5:
            return self.q_stable

        # Get recent errors
        errs = self.pred_errors[arm]
        mean_err = np.mean(errs)
        
        # Normalize error relative to threshold
        # If mean_err > threshold, we are in a non-stationary regime
        ratio = mean_err / self.error_threshold
        
        # Log-linear scaling of Q
        # q = q_base * (ratio ** adapt_gain)
        # Clamp
        q = self.q_stable * (max(ratio, 1e-3) ** self.adapt_gain)
        
        # Blend with mode-based Q
        if self.mode[arm] == 1:
            q = max(q, self.q_volatile)
        else:
            q = min(q, self.q_volatile)
            
        return max(self.q_min, min(self.q_max, q))

    def _update_regime(self, arm, innov_sq):
        """
        Update the regime (mode) based on innovation magnitude.
        Uses hysteresis to avoid frequent switching.
        """
        # Update EMA of squared innovations
        self._innov_ema[arm] = (1.0 - self._innov_alpha) * self._innov_ema[arm] + self._innov_alpha * innov_sq
        
        # Switching logic with hysteresis
        if self.mode[arm] == 0:  # Currently stable
            if innov_sq > self._switch_threshold_stable:
                self.mode[arm] = 1  # Switch to volatile
        else:  # Currently volatile
            if innov_sq < self._switch_threshold_volatile:
                self.mode[arm] = 0  # Switch to stable

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft reset: inflate covariance to encourage re-exploration
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # Uncertainty: context^T P context
            P_ctx = self.P[a] @ context
            var_term = context @ P_ctx
            if var_term < 0:
                var_term = 0.0
            bonus = self.alpha * np.sqrt(var_term)
            
            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.arm_counts[a]))
            alpha_t = max(self.alpha_min, self.alpha * (1.0 + self.round_num) ** (-0.1))
            
            scores.append(mu + alpha_t * count_factor * np.sqrt(var_term))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        # Decay alpha slightly to reduce exploration over time
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        # 1. Prediction Step (Random Walk)
        # Adapt Q based on history and mode
        q = self._adapt_q(arm, 0.0) # Pass 0, we update error buffer below
        
        # theta_pred = theta (Identity transition)
        # P_pred = P + q * I
        P_pred = self.P[arm] + q * np.eye(self.d)

        # 2. Update Step (Kalman Filter)
        # Observation: y = reward
        # H = context (1 x d)
        # R = observation noise variance. Assume 0.1 for Gaussian bandit.
        R = 0.1
        
        # Innovation
        y_hat = self.theta[arm] @ context
        innovation = reward - y_hat
        
        # Innovation Covariance
        S = context @ P_pred @ context + R
        
        # Kalman Gain
        if S > 1e-10:
            K = P_pred @ context / S
        else:
            K = np.zeros(self.d)
        
        # State Update
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update (Joseph form for stability)
        # P_new = (I - K H) P_pred (I - K H)^T + K R K^T
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R
        
        # Symmetrize
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # 3. Update Tracking Stats
        self.arm_counts[arm] += 1
        
        # Store prediction error (before update) for Q adaptation
        err_idx = self.pred_error_idx[arm]
        self.pred_errors[arm][err_idx] = abs(innovation)
        self.pred_error_idx[arm] = (err_idx + 1) % 10
        
        # Update regime based on innovation
        innov_sq = (innovation * innovation) / S
        self._update_regime(arm, innov_sq)
        
        # Check for reset
        self._maybe_reset(arm, innov_sq)

# EVOLVE-BLOCK-END
