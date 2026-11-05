"""Non-stationary linear contextual bandit with adaptive observation noise Kalman filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-arm Kalman filter with adaptive observation noise (R) that dynamically
    adjusts measurement noise variance based on standardized innovation magnitude.
    
    Structural change: instead of inflating process noise (Q) to handle non-stationarity,
    we adapt the observation noise (R) to down-weight potentially corrupted or outlier-heavy
    observations during abrupt regime shifts. This preserves the integrity of weight estimates
    while maintaining responsiveness to genuine signal changes.
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

        # Base observation noise
        self.R_base = 0.1
        self.R_max = 2.0  # Maximum R to allow when detecting outliers

        # Adaptive R parameters
        self._std_innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15
        self._r_threshold = 3.0  # Standardized innovation threshold for R adaptation
        self._r_scale = 2.0  # Scale factor for R increase

        # Process noise (fixed, small)
        self.Q = 3e-4

        # Exploration parameters
        self.alpha = 1.2
        self._alpha_min = 0.3

    def _adaptive_R(self, arm, innov_sq, denom):
        """
        Compute adaptive observation noise based on standardized innovation.
        
        If the standardized innovation (sqrt(innov_sq)) exceeds the threshold,
        we increase R to down-weight this observation, preserving theta integrity.
        """
        if self.count[arm] < 5:
            return self.R_base
        
        std_innov = np.sqrt(innov_sq) if innov_sq > 0 else 0.0
        ema_val = self._std_innov_ema[arm]
        
        # If recent innovations have been consistently large, increase R
        if std_innov > self._r_threshold:
            # Scale R up based on how much the innovation exceeds threshold
            excess = std_innov - self._r_threshold
            r_mult = 1.0 + self._r_scale * np.tanh(excess / 1.0)
            r = self.R_base * r_mult
            r = min(r, self.R_max)
        else:
            # Decay R back toward base if innovations are normal
            r = self.R_base
        
        return r

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

        # 1. Predict (Fixed small process noise)
        P_pred = self.P[a] + self.Q * np.eye(self.d)

        # 2. Calculate Innovation
        y = r - self.theta[a] @ x

        # 3. Compute preliminary R for innovation standardization
        R_prelim = self.R_base
        Px = P_pred @ x
        denom_prelim = x @ Px + R_prelim
        if denom_prelim < 1e-12:
            denom_prelim = 1e-12
        innov_sq_prelim = (y * y) / denom_prelim

        # 4. Adaptive R
        R_adapted = self._adaptive_R(a, innov_sq_prelim, denom_prelim)

        # 5. Update Step (Kalman, Joseph Form) with adapted R
        denom = x @ Px + R_adapted
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom

        # Update Theta
        self.theta[a] = self.theta[a] + K * y

        # Update P (Joseph Form for numerical stability)
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * R_adapted

        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0

        # Track standardized innovation for adaptive R
        innov_sq = (y * y) / denom
        std_innov = np.sqrt(innov_sq) if innov_sq > 0 else 0.0
        self._std_innov_ema[a] = (1.0 - self._innov_alpha) * self._std_innov_ema[a] + self._innov_alpha * std_innov

        self.count[a] += 1
# EVOLVE-BLOCK-END
