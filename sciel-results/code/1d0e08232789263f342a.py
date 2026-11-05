# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm Kalman Filter with Context-Scaled Process Noise.

    Structural change: Replaces the innovation-based adaptive Q with a context-scaled
    process noise model (Q_t = Q_base + lambda * ||x_t||^2). This hypothesizes that
    larger context magnitudes induce higher variance in the underlying weight drift
    or reward signal, allowing the filter to automatically increase uncertainty
    inflation for high-norm contexts without relying on lagged error signals.

    Refinements from inspirations:
    - Retains the robust Joseph form covariance update for numerical stability.
    - Keeps the 'reset' mechanism for catastrophic drift detection based on consecutive
      large innovations.
    - Maintains the count-scaled UCB exploration strategy with time decay.
    - Uses low observation noise (R=0.1) for sharper tracking.
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

        # Context-Scaled Process Noise parameters
        # Q_t = Q_base + Q_scale * ||x_t||^2
        self.Q_base = 1e-4
        self.Q_scale = 0.01

        # Observation Noise (R)
        # Lower R allows for more aggressive updates
        self.obs_noise = 0.1

        # Exploration parameters
        self.alpha = 1.2
        self._alpha_min = 0.3

        # Reset mechanism for catastrophic drift
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0  # Safety margin before 150s kill

    def _compute_Q(self, arm, context):
        """Compute context-scaled process noise."""
        # Use squared norm of context to scale process noise
        # This assumes that high-norm contexts are associated with higher volatility
        norm_sq = float(context @ context)
        q = self.Q_base + self.Q_scale * norm_sq
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
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
            
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

        # 1. Predict step: inflate covariance by context-scaled process noise
        Q = self._compute_Q(a, x)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # 2. Calculate Innovation
        predicted_reward = self.theta[a] @ x
        innovation = r - predicted_reward

        # 3. Update step: Kalman gain
        P_ctx = P_pred @ x
        denom = x @ P_ctx + self.obs_noise
        if denom < 1e-12:
            denom = 1e-12
        K = P_ctx / denom  # Kalman gain vector

        # Update state
        self.theta[a] = self.theta[a] + K * innovation

        # Update covariance using Joseph form for numerical stability
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self.obs_noise
        
        # Ensure symmetry
        self.P[a] = 0.5 * (P_new + P_new.T)

        # Check for reset based on normalized innovation squared
        innov_sq = (innovation * innovation) / denom
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
