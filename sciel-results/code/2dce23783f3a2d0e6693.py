"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with adaptive process noise, UCB exploration, and occasional resets.
    
    Key improvements over baseline:
    - Adaptive process noise via innovation EMA with tanh saturation.
    - Joseph-form covariance update for numerical stability.
    - Count-scaled UCB bonus that decays as we learn more about each arm.
    - Global alpha decay to reduce exploration over time.
    - Periodic soft-reset of covariance when prediction errors spike,
      allowing faster recovery from large weight shifts.
    - Tighter deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0  # safety margin before 120s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize covariance moderate to balance exploration and exploitation
        self.P = [np.eye(dim) * 2.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 1e-4
        self._Q_max = 0.02
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # EMA smoothing factor

        # Adaptive observation noise: estimate R from innovations
        self._R = 0.1
        self._R_ema = [0.1 for _ in range(n_arms)]
        self._R_alpha = 0.1
        self._R_min = 0.01
        self._R_max = 1.0

        # UCB parameter (adaptive)
        self.alpha = 1.0
        self._alpha_min = 0.2
        self._alpha_decay_rate = 500.0

        # Reset mechanism: track consecutive large innovations
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 4.0  # number of consecutive large innovations
        self._reset_cov_scale = 3.0  # scale to inflate covariance to (soft reset)

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        # Use a smoother, faster-responding mapping
        scaled = np.tanh(ema_val / 0.25)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _adaptive_R(self, arm):
        """Estimate observation noise from innovation statistics."""
        if self.count[arm] < 5:
            return self._R
        # Use the EMA of raw squared prediction errors (not normalized by S)
        # This gives a direct estimate of residual variance
        r_est = max(self._R_min, min(self._R_max, self._R_ema[arm]))
        return r_est

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, soft-inflate covariance."""
        if innov_sq > 3.5:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft inflation: add to existing covariance rather than replacing
            self.P[arm] = self.P[arm] + np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm using UCB with Kalman filter estimates."""
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ ctx
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            # Gentler decay: slower power-law decrease
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.05))
            
            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update Kalman filter for the selected arm with observed reward."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Prediction step (random walk model)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # Update step
        y = r - self.theta[a] @ x

        # Use adaptive R for this arm
        R_eff = self._adaptive_R(a)

        P_pred_x = P_pred @ x
        S = x @ P_pred_x + R_eff
        if S < 1e-12:
            S = 1e-12

        K = P_pred_x / S

        self.theta[a] = self.theta[a] + K * y

        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Update adaptive R estimate from raw prediction error
        raw_err_sq = y * y
        self._R_ema[a] = (1.0 - self._R_alpha) * self._R_ema[a] + self._R_alpha * raw_err_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
