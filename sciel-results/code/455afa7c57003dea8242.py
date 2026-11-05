"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-Arm Kalman Filter with Regime-Switching Process Noise.
    
    Maintains two Kalman filter states per arm:
    - 'stable' mode: low process noise, for gradual drift
    - 'volatile' mode: high process noise, for abrupt structural breaks
    
    Uses a lightweight HMM-like mechanism to blend between modes
    based on recent innovation magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 115s safety margin

        # Two covariance and mean states per arm
        # Mode 0: stable (low noise), Mode 1: volatile (high noise)
        self.theta = [[np.zeros(dim) for _ in range(2)] for _ in range(n_arms)]
        self.P = [[np.eye(dim) * 1.0 for _ in range(2)] for _ in range(n_arms)]

        # Process noise for each mode
        self.Q_stable = 0.005    # Low process noise
        self.Q_volatile = 0.5    # High process noise

        # Observation noise
        self.R = 1.0

        # HMM transition probabilities
        self.p_stay_stable = 0.995   # Probability of staying in stable mode
        self.p_stay_volatile = 0.90  # Probability of staying in volatile mode

        # Innovation threshold for switching
        self.innovation_threshold = 2.0

        # Track recent innovations for adaptive switching
        self.innovation_history = [np.zeros(10) for _ in range(n_arms)]
        self.innovation_idx = [0 for _ in range(n_arms)]

        # UCB parameter
        self.beta = 1.0

        # Blending weights (learned)
        self.w_stable = [1.0 for _ in range(n_arms)]  # Initial weight for stable mode

    def _check_deadline(self):
        return time.time() > self.deadline

    def _compute_innovation(self, context, arm, reward):
        """Compute innovation (prediction error) for the selected arm."""
        # Use the blended mean for innovation computation
        w = self.w_stable[arm]
        theta_blend = w * self.theta[arm][0] + (1 - w) * self.theta[arm][1]
        return reward - np.dot(theta_blend, context)

    def _update_mode_weights(self, arm, innovation):
        """Update mode blending weights based on innovation magnitude."""
        abs_innov = abs(innovation)
        
        # If innovation is large, shift weight toward volatile mode
        if abs_innov > self.innovation_threshold:
            self.w_stable[arm] *= 0.8  # Reduce weight on stable mode
        else:
            # If innovation is small, shift weight back toward stable mode
            self.w_stable[arm] = min(1.0, self.w_stable[arm] * 1.05 + 0.01)
        
        # Store innovation in history
        self.innovation_history[arm][self.innovation_idx[arm] % 10] = abs_innov
        self.innovation_idx[arm] += 1

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Compute blended mean and covariance
            w = self.w_stable[a]
            
            # Blend means
            theta_blend = w * self.theta[a][0] + (1 - w) * self.theta[a][1]
            
            # Blend covariances (approximate by weighted sum)
            P_blend = w * self.P[a][0] + (1 - w) * self.P[a][1]

            # Predict step: inflate covariance with process noise
            # Use blended process noise
            Q_blend = w * self.Q_stable + (1 - w) * self.Q_volatile
            norm_sq = np.dot(context, context)
            Q = Q_blend * (1 + 0.1 * norm_sq) * np.eye(self.d)
            
            P_pred = P_blend + Q

            # Compute score
            mean_reward = np.dot(theta_blend, context)
            var = np.dot(context, P_pred @ context)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)

            scores.append(mean_reward + self.beta * std)

        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Compute innovation
        innovation = self._compute_innovation(context, arm, reward)
        
        # Update mode weights
        self._update_mode_weights(arm, innovation)
        w = self.w_stable[arm]

        # Update both Kalman filters, but weight the updates
        for mode in range(2):
            # Get process noise for this mode
            Q_base = self.Q_stable if mode == 0 else self.Q_volatile
            norm_sq = np.dot(context, context)
            Q = Q_base * (1 + 0.1 * norm_sq) * np.eye(self.d)

            # Predicted covariance
            P_pred = self.P[arm][mode] + Q

            # Measurement update
            S = np.dot(context, P_pred @ context) + self.R
            if S < 1e-12:
                S = 1e-12

            K = (P_pred @ context) / S
            residual = reward - np.dot(self.theta[arm][mode], context)
            self.theta[arm][mode] = self.theta[arm][mode] + K * residual

            I_KH = np.eye(self.d) - np.outer(K, context)
            self.P[arm][mode] = I_KH @ P_pred
            self.P[arm][mode] = (self.P[arm][mode] + self.P[arm][mode].T) / 2.0

            # Weight the update: stable mode gets more weight when w is high
            # We already have separate states, so no additional weighting needed
# EVOLVE-BLOCK-END
