"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor for non-stationary rewards."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.alpha = 1.0  # exploration constant
        self.forgetting = 0.95  # adaptive forgetting factor
        self.forgetting_min = 0.85
        self.forgetting_max = 0.99
        self.error_ma = 0.0  # moving average of squared prediction error
        self._start_time = time.time()
        self._deadline = 115.0  # safety margin before 120s kill

    def _check_deadline(self):
        """Return True if we should stop exploring and return a safe default."""
        return (time.time() - self._start_time) > self._deadline

    def select(self, context):
        if self._check_deadline():
            # Return best-known arm or a safe default
            if hasattr(self, '_last_best_arm'):
                return self._last_best_arm
            return 0

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Solve A @ theta = b and A @ v = context
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                v = np.linalg.solve(self.A[a], context)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                v = np.zeros(self.d)
            mean = theta @ context
            variance = context @ v
            scores[a] = mean + self.alpha * np.sqrt(max(variance, 0.0))

        arm = int(np.argmax(scores))
        self._last_best_arm = arm
        return arm

    def update(self, context, arm, reward):
        # Compute prediction error for the selected arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            predicted = theta @ context
        except np.linalg.LinAlgError:
            predicted = 0.0

        error = reward - predicted
        sq_error = error * error

        # Update moving average of squared error
        self.error_ma = 0.9 * self.error_ma + 0.1 * sq_error

        # Adapt forgetting factor based on prediction error
        # High error -> lower forgetting (faster adaptation)
        # Low error -> higher forgetting (slower adaptation, more stability)
        if self.error_ma > 2.0:
            target_forgetting = self.forgetting_min
        elif self.error_ma > 0.5:
            target_forgetting = self.forgetting_min + \
                (self.forgetting_max - self.forgetting_min) * \
                (1.0 - (self.error_ma - 0.5) / 1.5)
        else:
            target_forgetting = self.forgetting_max

        # Smoothly adjust forgetting factor
        self.forgetting = 0.9 * self.forgetting + 0.1 * target_forgetting
        self.forgetting = max(self.forgetting_min, min(self.forgetting_max, self.forgetting))

        # Update with adaptive forgetting
        self.A[arm] = self.forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forgetting * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
