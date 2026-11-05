# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Dual-timescale Kalman filter for non-stationary linear contextual bandits.
    Improvements:
    1. Process noise (covariance inflation) is applied to ALL arms in select(),
       ensuring unvisited arms maintain high uncertainty and are explored.
    2. Adaptive process noise scaling based on innovation (prediction error).
    3. Joseph form covariance update for numerical stability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0  # Safety margin before 120s limit

        # Kalman filter state per arm
        # theta: current estimate of weight vector
        # P: covariance of theta estimate
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # Large initial uncertainty

        # Process noise parameters
        self.Q_base = 0.02  # Base process noise covariance scale (increased)
        self.sigma2 = 1.0   # Observation noise variance
        self.alpha = 0.15   # Adaptation rate for process noise scaling (faster adaptation)
        self.Q_floor = 0.005  # Minimum process noise to maintain exploration

        # Adaptive process noise scaling per arm
        self.pred_error = [0.0 for _ in range(n_arms)]
        self.Q_scale = [1.0 for _ in range(n_arms)]

        # UCB exploration coefficient (will decay over time)
        self.beta = 2.0
        self.beta_decay = 0.9995  # Slight decay to shift toward exploitation

        # Visit counts for calibration
        self.visits = [0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return the arm with highest mean estimate (exploit)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64).flatten()
        scores = []
        
        # Step 1: Inflate covariance for ALL arms to account for non-stationarity
        # This is critical: if we only inflate the selected arm in update(),
        # unselected arms' uncertainty freezes, leading to under-exploration.
        # Use time-scaled process noise: increases as horizon progresses
        time_scale = 1.0 + 0.1 * self.t  # Gradually increase process noise
        for a in range(self.n):
            Q = self.Q_base * self.Q_scale[a] * time_scale * np.eye(self.d)
            self.P[a] = self.P[a] + Q

        # Step 2: Compute scores with inflated covariance
        # Decay beta over time to shift toward exploitation
        current_beta = self.beta * (self.beta_decay ** self.t)
        for a in range(self.n):
            mean = self.theta[a] @ context
            # UCB bonus based on Kalman filter covariance
            var = context @ self.P[a] @ context
            bonus = current_beta * np.sqrt(var)
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        self.visits[arm] += 1

        # Note: Covariance for this arm was already inflated in select().
        # We use the current P[arm] as P_pred.
        P_pred = self.P[arm]

        # Update step: Kalman gain
        # K = P_pred @ x / (x^T P_pred x + sigma2)
        Px = P_pred @ context
        denom = context @ Px + self.sigma2
        K = Px / denom  # shape: (d,)

        # Innovation (prediction error)
        y_pred = self.theta[arm] @ context
        innovation = reward - y_pred

        # Update theta and P
        self.theta[arm] = self.theta[arm] + K * innovation
        # Joseph form for numerical stability
        I_Kx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_Kx @ P_pred @ I_Kx.T + self.sigma2 * np.outer(K, K)

        # Adapt process noise scaling based on prediction error
        # If innovation is large relative to expected, increase Q_scale
        expected_innovation_std = np.sqrt(denom)
        error_ratio = (innovation ** 2) / (self.sigma2 + 1e-6)
        # Exponential moving average of error ratio
        self.pred_error[arm] = (1 - self.alpha) * self.pred_error[arm] + self.alpha * error_ratio
        # Scale Q: higher when error is high, with floor to maintain exploration
        self.Q_scale[arm] = max(1.0 + self.pred_error[arm], self.Q_floor / self.Q_base)
        # Cap to prevent instability
        self.Q_scale[arm] = min(self.Q_scale[arm], 100.0)

        self.t += 1
# EVOLVE-BLOCK-END
