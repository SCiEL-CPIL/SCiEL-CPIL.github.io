"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with non-stationary handling and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin
        self._recent_errors = []
        self._base_lambda = 0.97
        self._min_lambda = 0.85
        self._max_lambda = 0.995
        self._theta_cache = [None] * n_arms
        self._inv_cache = [None] * n_arms
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda
        
        recent_mean_error = np.mean(self._recent_errors[-20:])
        # Higher error -> lower lambda (faster forgetting)
        # Map error range roughly [0, 2] to lambda range [max, min]
        # Use a simple linear mapping with clamping
        error_norm = np.clip(recent_mean_error / 2.0, 0.0, 1.0)
        lambda_val = self._max_lambda - error_norm * (self._max_lambda - self._min_lambda)
        return lambda_val

    def _update_matrices(self, arm, context, reward):
        """Update A and b with forgetting factor, and clear caches."""
        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        # Clear cache for this arm
        self._theta_cache[arm] = None
        self._inv_cache[arm] = None

    def select(self, context):
        self._round += 1
        # Deadline guard: switch to greedy if near limit
        if time.time() > self._deadline - 5.0:
            # Greedy: pick arm with highest predicted mean
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                if self._theta_cache[a] is None:
                    inv = np.linalg.inv(self.A[a])
                    self._inv_cache[a] = inv
                    self._theta_cache[a] = inv @ self.b[a]
                theta = self._theta_cache[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = []
        for a in range(self.n):
            if self._theta_cache[a] is None:
                inv = np.linalg.inv(self.A[a])
                self._inv_cache[a] = inv
                self._theta_cache[a] = inv @ self.b[a]
            
            theta = self._theta_cache[a]
            inv = self._inv_cache[a]
            
            mean = theta @ context
            # Uncertainty bonus
            var = context @ inv @ context
            if var < 0:
                var = 0.0
            bonus = 1.0 * np.sqrt(var)
            scores.append(mean + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Compute prediction error for adaptive lambda
        if self._theta_cache[arm] is not None:
            predicted = self._theta_cache[arm] @ context
            error = reward - predicted
        else:
            inv = np.linalg.inv(self.A[arm])
            self._inv_cache[arm] = inv
            self._theta_cache[arm] = inv @ self.b[arm]
            predicted = self._theta_cache[arm] @ context
            error = reward - predicted
        
        self._recent_errors.append(error)
        if len(self._recent_errors) > 100:
            self._recent_errors.pop(0)
        
        self._update_matrices(arm, context, reward)
# EVOLVE-BLOCK-END
