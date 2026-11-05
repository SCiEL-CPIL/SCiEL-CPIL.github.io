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
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize covariance large to encourage exploration early
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Adaptive process noise
        self.Q_base = 0.01
        self.Q_scale = [1.0 for _ in range(n_arms)]  # per-arm adaptive scale
        
        # Exploration coefficient
        self.alpha = 1.0
        
        # Track prediction errors for adaptive Q
        self.pred_errors = [0.0 for _ in range(n_arms)]
        self.error_counts = [0 for _ in range(n_arms)]
        
        # Counts for each arm
        self.counts = [0 for _ in range(n_arms)]
        
        # Global trend estimator: linear model of context -> baseline reward
        # This captures common environmental drift shared across arms
        self.trend_theta = np.zeros(dim)
        self.trend_P = np.eye(dim) * 10.0
        self.trend_Q = 0.05 * np.eye(dim)  # slower adaptation for trend
        self.trend_R = 1.0  # measurement noise for trend
        self.trend_count = 0
        self.trend_pred_errors = 0.0
        self.trend_error_count = 0

    def select(self, context):
        """Select arm using UCB with Kalman filter estimates and trend detrending."""
        self.t += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return best-known arm (highest mean reward estimate)
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        # Estimate global trend component
        trend_mu = self.trend_theta @ context
        
        scores = []
        for a in range(self.n):
            # Predicted arm-specific reward (detrended)
            mu_arm = self.theta[a] @ context
            
            # Total predicted reward = trend + arm-specific
            mu = trend_mu + mu_arm
            
            # Uncertainty: combine trend and arm uncertainty
            Px = self.P[a] @ context
            var_arm = context @ Px
            if var_arm < 0:
                var_arm = 0.0
            
            # Trend uncertainty
            Pxt = self.trend_P @ context
            var_trend = context @ Pxt
            if var_trend < 0:
                var_trend = 0.0
            
            variance = var_arm + var_trend
            std = np.sqrt(variance)
            
            # Adaptive exploration: reduce alpha as we learn more
            # Use sqrt(1 + count) decay for exploration
            count_factor = 1.0 / (1.0 + np.sqrt(self.counts[a]))
            exploration = self.alpha * std * count_factor
            
            scores.append(mu + exploration)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update Kalman filter for the selected arm with observed reward, using trend detrending."""
        self.counts[arm] += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return
        
        a = arm
        x = context
        r = reward
        
        # First, update the global trend estimator
        # Predict
        trend_P_pred = self.trend_P + self.trend_Q
        
        # Innovation for trend: y_trend = r - trend_theta^T x
        y_trend = r - self.trend_theta @ x
        
        # Innovation covariance
        S_trend = x @ trend_P_pred @ x + self.trend_R
        
        # Kalman gain for trend
        K_trend = trend_P_pred @ x / S_trend
        
        # Update trend theta
        self.trend_theta = self.trend_theta + K_trend * y_trend
        
        # Update trend P (Joseph form)
        IK_trend = np.eye(self.d) - np.outer(K_trend, x)
        self.trend_P = IK_trend @ trend_P_pred @ IK_trend.T + self.trend_Q
        
        # Ensure symmetry
        self.trend_P = (self.trend_P + self.trend_P.T) / 2.0
        
        # Track trend prediction error
        self.trend_pred_errors += abs(y_trend)
        self.trend_error_count += 1
        
        # Decay trend error tracking
        if self.trend_error_count > 50:
            self.trend_pred_errors *= 0.95
            self.trend_error_count = int(self.trend_error_count * 0.95)
        
        # Now update the arm-specific model using detrended reward
        # Detrended reward: r_detrended = r - trend_theta^T x
        r_detrended = r - self.trend_theta @ x
        
        # Adaptive process noise for arm
        if self.error_counts[a] > 0:
            avg_error = self.pred_errors[a] / self.error_counts[a]
            # Scale Q based on error magnitude
            self.Q_scale[a] = 1.0 + avg_error * 2.0
            # Cap the scale
            self.Q_scale[a] = min(self.Q_scale[a], 10.0)
        
        Q = self.Q_base * self.Q_scale[a] * np.eye(self.d)
        
        # Predict
        P_pred = self.P[a] + Q
        
        # Update step (measurement: r_detrended = theta^T x + noise)
        # Innovation: y = r_detrended - theta^T x
        y = r_detrended - self.theta[a] @ x
        
        # Innovation covariance: S = x^T P_pred x + R
        # Assume measurement noise R = 1.0
        S = x @ P_pred @ x + 1.0
        
        # Kalman gain: K = P_pred x / S
        K = P_pred @ x / S
        
        # Update theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P (Joseph form for numerical stability)
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q  # Add Q back for process noise
        
        # Ensure P is symmetric
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        
        # Track prediction error for adaptive Q
        self.pred_errors[a] += abs(y)
        self.error_counts[a] += 1
        
        # Decay error tracking
        if self.error_counts[a] > 50:
            self.pred_errors[a] *= 0.95
            self.error_counts[a] = int(self.error_counts[a] * 0.95)

# EVOLVE-BLOCK-END
