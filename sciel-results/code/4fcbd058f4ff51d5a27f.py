"""Non-stationary linear contextual bandit with adaptive Kalman filter and drift detection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-arm adaptive Kalman filter with continuous process noise adaptation
    and catastrophic drift detection via covariance reset.
    
    Structural change from parent: replaces discrete regime-switching with
    continuous tanh-saturated EMA of squared innovations for Q adaptation,
    plus a reset mechanism for catastrophic drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 110.0  # Safety margin before 120s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage initial exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive Process Noise (Q) parameters
        self.Q_base = 3e-4
        self.Q_max = 0.05
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12

        # Observation Noise
        self.obs_noise = 0.1

        # Exploration parameters
        self.alpha = 1.2
        self._alpha_min = 0.3

        # Reset mechanism for catastrophic drift
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self.Q_base
        ema_val = self._innov_ema[arm]
        scaled = np.tanh(ema_val / 0.4)
        q = self.Q_base + (self.Q_max - self.Q_base) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)

        if self._consec_large[arm] >= self._reset_threshold:
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context

            # Uncertainty
            Px = self.P[a] @ context
            var = context @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)

            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))

            score = mean + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = float(reward)

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict (Process Noise)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # 2. Calculate Innovation
        y = r - self.theta[a] @ x

        # 3. Update Step (Kalman, Joseph Form)
        sigma2 = self.obs_noise
        Px = P_pred @ x
        denom = x @ Px + sigma2
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom

        # Update Theta
        self.theta[a] = self.theta[a] + K * y

        # Update P (Joseph Form for numerical stability)
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * sigma2

        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0

        # Track prediction error for adaptive Q
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
