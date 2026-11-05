"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with adaptive process noise and UCB exploration.
    
    Key ideas:
    - Model each arm's weight vector theta_a as a random walk.
    - Use Kalman filter updates for fast adaptation to changing weights.
    - Adaptive process noise: increase Q when prediction errors are large.
    - UCB exploration with time-decayed confidence to balance exploitation/exploration.
    - Deadline guard to ensure we finish within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # safety margin before 120s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize covariance large to encourage exploration early
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 5e-4
        self._Q_max = 0.05
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # EMA smoothing factor

        # Observation noise
        self._R = 0.1

        # Thompson Sampling: no fixed alpha needed, but keep for fallback
        self.alpha = 1.0
        self._alpha_min = 0.3
        self._alpha_decay_rate = 300.0
        # Adaptive covariance inflation factor (smoothed)
        self._inflation = [1.0 for _ in range(n_arms)]
        self._inflation_alpha = 0.1  # EMA smoothing for inflation
        self._inflation_min = 0.5
        self._inflation_max = 3.0

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        # Smooth tanh saturation to avoid extreme Q values
        ema_val = self._innov_ema[arm]
        # Map [0, ~2] to [0, 1] via tanh for smoothness
        scaled = np.tanh(ema_val / 0.5)  # saturates around ema_val ~ 1.5
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def select(self, context):
        """Select arm using Thompson Sampling with Kalman filter posteriors."""
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest mean estimate
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted mean
            mu = self.theta[a] @ ctx
            # Predicted variance with adaptive inflation
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            # Apply adaptive inflation to encourage exploration when non-stationary
            var_inflated = var * self._inflation[a]
            std = np.sqrt(var_inflated)
            
            # Thompson Sampling: sample from posterior N(mu, var)
            # Use the standardized innovation to determine exploration level
            if self.count[a] > 2:
                # Sample from the posterior
                score = mu + np.random.randn() * std
            else:
                # Early on, use UCB-like bonus for exploration
                count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
                alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
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

        # Update step (measurement: r = theta^T x + noise)
        # Innovation: y = r - theta^T x
        y = r - self.theta[a] @ x

        # Innovation covariance: S = x^T P_pred x + R
        P_pred_x = P_pred @ x
        S = x @ P_pred_x + self._R
        if S < 1e-12:
            S = 1e-12

        # Kalman gain: K = P_pred x / S
        K = P_pred_x / S

        # Update theta
        self.theta[a] = self.theta[a] + K * y

        # Update P (Joseph form for numerical stability)
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)

        # Ensure P is symmetric
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q (EMA of squared innovation)
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Update adaptive inflation based on standardized innovation magnitude
        # Large innovations suggest non-stationarity -> inflate covariance more
        standardized_innov = abs(y) / np.sqrt(S) if S > 1e-12 else 0.0
        # Map standardized innovation to inflation factor in [min, max]
        # Use tanh for smooth saturation
        target_inflation = self._inflation_min + (self._inflation_max - self._inflation_min) * np.tanh(standardized_innov / 2.0)
        # Smooth update
        self._inflation[a] = (1.0 - self._inflation_alpha) * self._inflation[a] + self._inflation_alpha * target_inflation

        self.count[a] += 1
# EVOLVE-BLOCK-END
