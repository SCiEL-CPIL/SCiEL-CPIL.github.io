# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Hybrid Adaptive Kalman Filter with Joseph Form and Reset Mechanism.

    Structural change: Combines the robust state-space modeling of the Kalman filter
    (from current program) with the numerical stability of the Joseph form covariance
    update and the 'reset' mechanism for catastrophic drift detection (from inspirations).
    
    Refinements:
    - Uses tanh-saturated EMA of squared innovations for smooth, bounded adaptive process noise (Q).
    - Reduces observation noise (R) to 0.1 to allow sharper tracking, consistent with high-scoring inspirations.
    - Implements a decaying exploration coefficient scaled by arm visit count.
    - Includes a deadline guard to ensure completion within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage initial exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive Process Noise (Q) parameters
        # Base Q is small, but can grow up to Q_max if innovations are large
        self.Q_base = 5e-4
        self.Q_max = 0.06
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.08  # Slower EMA for more stable Q adaptation
        self._innov_ema_sq = [0.0 for _ in range(n_arms)]  # EMA of squared innovations for better scaling

        # Observation Noise (R)
        # Lower R (0.1) allows for more aggressive updates, assuming rewards are relatively low noise
        self.obs_noise = 0.1
        self._obs_noise_ema = 0.1  # Adaptive observation noise estimate
        self._obs_noise_alpha = 0.05  # Slow adaptation for R

        # Exploration parameters
        self.alpha = 1.4
        self._alpha_min = 0.2
        self._explore_decay = 0.08  # Slower decay for sustained exploration

        # Reset mechanism for catastrophic drift
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0  # Safety margin before 150s kill

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self.Q_base
        
        ema_val = self._innov_ema[arm]
        # Map EMA to [0, 1] range using tanh to ensure bounded Q
        scaled = np.tanh(ema_val / 0.4)
        q = self.Q_base + (self.Q_max - self.Q_base) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance to encourage re-exploration."""
        # Threshold for 'large' innovation (squared)
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

        # Deadline check: if close to deadline, exploit best estimate
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            
            # Uncertainty: sqrt(x^T P x)
            P_ctx = self.P[a] @ context
            variance = context @ P_ctx
            if variance < 0:
                variance = 0.0
            std = np.sqrt(variance)

            # Adaptive exploration: reduce bonus as we learn more
            # Count factor reduces exploration for well-visited arms
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            # Time decay: exploration coefficient decreases slowly over time
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-self._explore_decay))
            
            score = mean + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        a = arm
        x = context
        r = reward

        # Deadline guard: skip heavy computation if near deadline
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict step: inflate covariance by adaptive process noise
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # 2. Calculate Innovation
        predicted_reward = self.theta[a] @ x
        innovation = r - predicted_reward

        # 3. Update step: Kalman gain with adaptive observation noise
        P_ctx = P_pred @ x
        # Use adaptive observation noise
        R = self._obs_noise_ema
        denom = x @ P_ctx + R
        if denom < 1e-12:
            denom = 1e-12
        K = P_ctx / denom  # Kalman gain vector

        # Update state
        self.theta[a] = self.theta[a] + K * innovation

        # Update covariance using Joseph form for numerical stability
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * R
        
        # Ensure symmetry
        self.P[a] = 0.5 * (P_new + P_new.T)

        # Track error for adaptive Q
        # Use normalized innovation squared
        innov_sq = (innovation * innovation) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq
        self._innov_ema_sq[a] = (1.0 - self._innov_alpha) * self._innov_ema_sq[a] + self._innov_alpha * innov_sq
        
        # Adapt observation noise estimate
        if self.count[a] > 5:
            self._obs_noise_ema = (1.0 - self._obs_noise_alpha) * self._obs_noise_ema + self._obs_noise_alpha * innov_sq
            # Clamp to reasonable range
            self._obs_noise_ema = np.clip(self._obs_noise_ema, 0.01, 1.0)

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
