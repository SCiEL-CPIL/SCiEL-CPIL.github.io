"""Adaptive non-stationary LinUCB with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with dynamic forgetting factor and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 115.0  # 5s safety margin for 120s limit
        self._recent_errors = []
        self._max_error_window = 50
        self._base_lambda = 0.98
        self._min_lambda = 0.90
        self._max_lambda = 0.995
        self._alpha = 1.0  # exploration parameter
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda

        # Use exponential moving average of errors for stability
        recent = self._recent_errors[-self._max_error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Normalize error to [0, 1] range using a sigmoid-like mapping
        # Higher error -> lower lambda (faster forgetting)
        norm_err = np.tanh(mean_err / (0.5 + std_err + 1e-8))
        norm_err = (norm_err + 1.0) / 2.0  # map to [0, 1]

        # Interpolate lambda: high error -> min_lambda, low error -> max_lambda
        lambda_val = self._max_lambda - norm_err * (self._max_lambda - self._min_lambda)
        return float(np.clip(lambda_val, self._min_lambda, self._max_lambda))

    def select(self, context):
        self._round += 1
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: switch to greedy if near deadline
        if time.time() > self._deadline - 2.0:
            scores = []
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        lam = self._get_lambda()
        self._alpha = 1.0 + 0.5 * np.sqrt(self._round)  # adaptive exploration

        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                inv = np.eye(self.d)
            theta = inv @ self.b[a]
            mu = theta @ context
            # UCB bonus
            bonus = self._alpha * np.sqrt(np.maximum(context @ inv @ context, 0.0))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Compute prediction error before update
        try:
            inv = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            inv = np.eye(self.d)
        theta = inv @ self.b[arm]
        predicted = theta @ context
        error = reward - predicted

        # Store error for adaptive lambda
        self._recent_errors.append(abs(error))
        if len(self._recent_errors) > self._max_error_window:
            self._recent_errors.pop(0)

        # Update with forgetting
        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
