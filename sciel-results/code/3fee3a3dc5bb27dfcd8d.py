# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm Kalman filter with adaptive process noise for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._deadline = time.time() + 110.0  # safety margin before 120s

        # Per-arm state: theta (weight vector), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]

        # Adaptive process noise per arm
        self.Q_base = 0.01
        self.residuals = [[] for _ in range(n_arms)]
        self.residual_window = 20

        # Regularization / observation noise
        self.R = 0.5  # observation noise variance

        # Track best arm for deadline fallback
        self._best_arm = 0
        self._cumulative_reward = [0.0] * n_arms
        self._arm_counts = [0] * n_arms

    def _adaptive_Q(self, arm):
        """Adapt process noise based on recent residuals."""
        res = self.residuals[arm]
        if len(res) < 5:
            return self.Q_base
        recent = res[-self.residual_window:]
        mean_r = np.mean(recent)
        var_r = np.var(recent)
        # Increase Q if residuals are large or trending
        drift_signal = np.abs(mean_r) + np.sqrt(var_r)
        q = self.Q_base * (1.0 + 2.0 * drift_signal)
        return min(q, 1.0)  # cap to avoid instability

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return self._best_arm

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Prediction: theta @ context
            mu = self.theta[a] @ context
            # Uncertainty: sqrt(context @ P @ context)
            sigma = np.sqrt(max(0.0, context @ self.P[a] @ context))
            # UCB with adaptive exploration
            beta = 1.0 + 0.5 * np.log(1.0 + self.t + 1)
            scores[a] = mu + beta * sigma

        arm = int(np.argmax(scores))
        self._best_arm = arm
        return arm

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Store residual for adaptive Q
        predicted = self.theta[arm] @ context
        residual = reward - predicted
        self.residuals[arm].append(residual)
        if len(self.residuals[arm]) > self.residual_window:
            self.residuals[arm].pop(0)

        # Update cumulative stats for deadline fallback
        self._cumulative_reward[arm] += reward
        self._arm_counts[arm] += 1
        if self._arm_counts[arm] > 1:
            # Update best arm based on average reward
            avg = self._cumulative_reward[arm] / self._arm_counts[arm]
            if avg > (self._cumulative_reward[self._best_arm] / max(1, self._arm_counts[self._best_arm])):
                self._best_arm = arm

        # Kalman filter update
        A = self.P[arm]
        b = context
        r = self.R

        # Innovation
        innovation = reward - self.theta[arm] @ context
        S = b @ A @ b + r  # innovation variance

        # Kalman gain
        K = A @ b / S  # (d,1) -> (d,)

        # State update
        self.theta[arm] = self.theta[arm] + K * innovation

        # Covariance update (Joseph form for stability)
        IK = np.eye(self.d) - np.outer(K, b)
        self.P[arm] = IK @ A @ IK.T + r * np.outer(K, K)

        # Add process noise (adaptive)
        q = self._adaptive_Q(arm)
        self.P[arm] += q * np.eye(self.d)

        # Symmetrize to prevent drift
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(self.P[arm])
        eigvals = np.maximum(eigvals, 1e-8)
        self.P[arm] = eigvecs @ np.diag(eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
