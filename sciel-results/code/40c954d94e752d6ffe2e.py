"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with non-stationary handling and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Initialize matrices and vectors for each arm
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time tracking for deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin before 150s kill
        
        # Adaptive forgetting factor parameters
        self._base_lambda = 0.95
        self._min_lambda = 0.85
        self._max_lambda = 0.99
        self._recent_errors = []
        self._error_window = 20
        
        # Exploration parameters
        self._alpha = 1.0
        self._alpha_min = 0.5
        self._alpha_max = 2.0
        
        # Track round count
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < self._error_window:
            return self._base_lambda
        
        recent_mean_error = np.mean(self._recent_errors[-self._error_window:])
        # Higher error -> lower lambda (faster forgetting)
        # Scale error to lambda range
        # Assume typical error range is [0, 2]
        normalized_error = min(1.0, max(0.0, recent_mean_error / 2.0))
        lambda_val = self._max_lambda - normalized_error * (self._max_lambda - self._min_lambda)
        return lambda_val

    def _get_alpha(self):
        """Adapt exploration coefficient based on performance."""
        if len(self._recent_errors) < self._error_window:
            return self._alpha
        
        recent_mean_error = np.mean(self._recent_errors[-self._error_window:])
        # Higher error -> more exploration
        normalized_error = min(1.0, max(0.0, recent_mean_error / 2.0))
        alpha = self._alpha_min + normalized_error * (self._alpha_max - self._alpha_min)
        return alpha

    def select(self, context):
        self._round += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline - if we're running out of time, fall back to greedy
        if time.time() > self._deadline - 5.0:
            # Greedy fallback: pick arm with highest mean estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    inv = np.linalg.inv(self.A[a] + 0.001 * np.eye(self.d))
                    theta = inv @ self.b[a]
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    continue
            return best_arm

        lambda_val = self._get_lambda()
        alpha = self._get_alpha()
        
        scores = []
        for a in range(self.n):
            try:
                # Add small regularization for numerical stability
                A_reg = self.A[a] + 0.001 * np.eye(self.d)
                inv = np.linalg.inv(A_reg)
                theta = inv @ self.b[a]
                mean_est = theta @ context
                var_est = context @ inv @ context
                ucb = mean_est + alpha * np.sqrt(var_est)
                scores.append(ucb)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Track prediction error for adaptive lambda
        try:
            A_reg = self.A[arm] + 0.001 * np.eye(self.d)
            inv = np.linalg.inv(A_reg)
            theta = inv @ self.b[arm]
            predicted = theta @ context
            error = abs(reward - predicted)
            self._recent_errors.append(error)
            if len(self._recent_errors) > 100:
                self._recent_errors = self._recent_errors[-100:]
        except np.linalg.LinAlgError:
            pass

        # Update with forgetting factor
        lambda_val = self._get_lambda()
        self.A[arm] = lambda_val * self.A[arm] + np.outer(context, context)
        self.b[arm] = lambda_val * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
