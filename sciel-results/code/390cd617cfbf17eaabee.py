# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Kalman-filter-based non-stationary linear contextual bandit with global trend removal."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 130.0  # 20s safety margin

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Global trend estimator (scalar or low-dim)
        self.trend_A = np.eye(dim) * 10.0
        self.trend_b = np.zeros(dim)
        self.trend_theta = np.zeros(dim)
        self.trend_count = 0

        # Adaptive process noise parameters
        self.base_q = 0.01
        self.q_multiplier = 1.0
        self.error_history = []

        # Exploration coefficient
        self.alpha = 1.0

    def _get_trend(self, context):
        """Get the current trend estimate for a context."""
        return self.trend_theta @ context

    def _update_trend(self, context, reward):
        """Update the global trend estimator."""
        self.trend_A = self.trend_A + np.outer(context, context)
        self.trend_b = self.trend_b + reward * context
        self.trend_count += 1
        # Regularized inverse
        reg = 1e-6 * np.eye(self.d)
        self.trend_theta = np.linalg.solve(self.trend_A + reg, self.trend_b)

    def _adapt_process_noise(self, prediction_error):
        """Adaptively increase process noise when errors are large."""
        self.error_history.append(prediction_error)
        if len(self.error_history) > 20:
            self.error_history.pop(0)
        
        if len(self.error_history) >= 5:
            mean_err = np.mean(self.error_history)
            # Scale process noise based on recent error magnitude
            self.q_multiplier = 1.0 + 2.0 * np.clip(mean_err / 0.5, 0, 3)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Return the arm with highest mean estimate
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        trend = self._get_trend(context)

        scores = []
        for a in range(self.n):
            # Prediction: theta_a @ context - trend (de-trended)
            pred = self.theta[a] @ context - trend
            # Uncertainty: sqrt(context^T P_a context)
            P_ctx = self.P[a] @ context
            uncertainty = np.sqrt(max(P_ctx @ context, 0))
            score = pred + self.alpha * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Update global trend
        self._update_trend(context, reward)
        trend = self._get_trend(context)

        # De-trended reward
        detrended_reward = reward - trend

        # Kalman filter update for the selected arm
        # Prediction step (process noise)
        self._adapt_process_noise(
            abs(detrended_reward - (self.theta[arm] @ context - trend))
        )
        q = self.base_q * self.q_multiplier
        self.P[arm] = self.P[arm] + q * np.eye(self.d)

        # Update step
        # K = P H^T (H P H^T + R)^-1, where H = context, R = 1 (Gaussian noise)
        P_ctx = self.P[arm] @ context
        S = P_ctx @ context + 1.0  # innovation variance
        K = P_ctx / S  # Kalman gain

        # Innovation
        pred = self.theta[arm] @ context - trend
        innovation = detrended_reward - pred

        # Update theta
        self.theta[arm] = self.theta[arm] + K * innovation
        # Update P (Joseph form for stability)
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ self.P[arm] @ I_KH.T + K @ K.T * 1.0

        self.count[arm] += 1

        # Adaptive alpha: reduce exploration as confidence grows
        total_count = sum(self.count)
        if total_count > 50:
            self.alpha = max(0.3, 1.0 - 0.5 * np.log(total_count) / np.log(100))
# EVOLVE-BLOCK-END
