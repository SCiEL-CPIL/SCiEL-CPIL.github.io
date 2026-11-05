"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandit."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Store posterior covariance P and mean theta directly
        self.P = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Kalman filter parameters
        self.R = 0.1  # Observation noise
        self.Q_base = 0.01  # Base process noise
        self.Q_min = 0.001
        self.Q_max = 0.5
        
        # Adaptive process noise
        self.error_ema = 0.0
        self.error_alpha = 0.1
        
        # Exploration
        self.C = 0.3
        self.round_count = 0
        self._start_time = time.time()
        self._deadline = 140.0

    def _elapsed(self):
        return time.time() - self._start_time

    def select(self, context):
        if self._elapsed() > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))
        
        scores = []
        for a in range(self.n):
            pred = self.theta[a] @ context
            pred_var = context @ self.P[a] @ context
            scores.append(pred + self.C * np.sqrt(max(pred_var, 0.0)))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        x = context
        P = self.P[arm]
        theta = self.theta[arm]
        
        # Kalman filter measurement update
        pred = theta @ x
        innovation = reward - pred
        S = x @ P @ x + self.R
        K = (P @ x) / S  # Kalman gain, shape (d,)
        
        # Update mean
        theta_new = theta + K * innovation
        
        # Update covariance (Joseph form for stability)
        P_new = P - np.outer(K, x @ P)
        P_new = 0.5 * (P_new + P_new.T)
        
        # Prediction step: add process noise (random walk drift)
        if self.round_count > 5:
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * abs(innovation)
            norm_err = min(self.error_ema / (self.R + 0.1), 1.0)
            Q = self.Q_min + (self.Q_max - self.Q_min) * norm_err
        else:
            Q = self.Q_base
        
        # Add isotropic process noise to covariance
        P_new = P_new + Q * np.eye(self.d)
        
        # Store updated values
        self.theta[arm] = theta_new
        self.P[arm] = P_new
# EVOLVE-BLOCK-END
