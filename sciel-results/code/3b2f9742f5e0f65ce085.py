"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with non-stationary handling and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin
        self._recent_errors = []
        self._base_lambda = 0.97
        self._min_lambda = 0.90
        self._max_lambda = 0.995

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda
        
        recent_mean_error = np.mean(self._recent_errors[-20:])
        # Higher error -> faster forgetting (lower lambda)
        # Map error range [0, 2] to lambda range [max, min]
        error_norm = np.clip(recent_mean_error / 2.0, 0, 1)
        lam = self._max_lambda - error_norm * (self._max_lambda - self._min_lambda)
        return float(lam)

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() >= self._deadline

    def select(self, context):
        # If deadline is near, use greedy selection for speed
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    inv = np.linalg.inv(self.A[a])
                    theta = inv @ self.b[a]
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    continue
            return int(best_arm)
        
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                pred = theta @ context
                ucb = 0.5 * np.sqrt(context @ inv @ context)
                scores.append(pred + ucb)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track prediction error for adaptive forgetting
        try:
            inv = np.linalg.inv(self.A[arm])
            theta = inv @ self.b[arm]
            pred = theta @ context
            error = abs(reward - pred)
            self._recent_errors.append(error)
            if len(self._recent_errors) > 50:
                self._recent_errors.pop(0)
        except np.linalg.LinAlgError:
            pass

        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
