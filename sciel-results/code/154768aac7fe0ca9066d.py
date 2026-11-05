# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 115.0  # safety margin before 120s kill

        # Per-arm Kalman filter state
        # State: theta (dim), P (dim x dim) covariance
        # Process noise Q: random walk on theta
        # Observation noise R: scalar

        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # initial covariance (optimistic)

        # Hyperparameters
        self.R = 1.0  # observation noise variance
        self.Q_base = 0.05  # base process noise per step
        self.alpha = 0.1  # adaptation rate for Q scaling
        self.beta = 1.0  # exploration bonus coefficient

        # Track innovation history for adaptive Q
        self.innovation_history = [0.0 for _ in range(n_arms)]

    def _kalman_predict(self, arm):
        """Predict step: inflate covariance by process noise."""
        # Adaptive Q: scale by recent innovation magnitude
        innov_scale = max(1.0, self.innovation_history[arm])
        Q = self.Q_base * innov_scale * np.eye(self.d)
        self.P[arm] = self.P[arm] + Q

    def _kalman_update(self, arm, context, reward):
        """Update step: incorporate observation."""
        x = context
        z = reward

        # Predicted mean and variance
        mu_pred = self.theta[arm] @ x
        S = x @ self.P[arm] @ x + self.R  # innovation variance

        # Innovation
        y = z - mu_pred

        # Store innovation magnitude for adaptive Q
        self.innovation_history[arm] = 0.9 * self.innovation_history[arm] + 0.1 * (y ** 2)

        # Kalman gain
        if S > 1e-12:
            K = (self.P[arm] @ x) / S
        else:
            K = np.zeros(self.d)

        # Update state
        self.theta[arm] = self.theta[arm] + K * y
        # Joseph form for numerical stability
        I_Kx = np.eye(self.d) - np.outer(K, x)
        self.P[arm] = I_Kx @ self.P[arm] @ I_Kx.T + K @ K.T * self.R

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        x = context
        scores = []
        for a in range(self.n):
            # Predict
            self._kalman_predict(a)

            # Score: mean + exploration bonus
            mu = self.theta[a] @ x
            var = x @ self.P[a] @ x
            bonus = self.beta * np.sqrt(var)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() > self.deadline:
            return

        x = context
        z = reward

        # The prediction was already done in select for this arm,
        # but we need to be careful: select predicted ALL arms.
        # Actually, in select we called _kalman_predict for every arm.
        # So here we just do the update.
        self._kalman_update(arm, x, z)
# EVOLVE-BLOCK-END
