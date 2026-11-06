"""Non-stationary linear contextual bandit with adaptive forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    LinUCB with adaptive forgetting factor and adaptive exploration.
    
    Uses per-arm linear models with a forgetting mechanism that adapts
    based on prediction error. Higher error increases the forgetting rate
    to track non-stationarity faster, and increases exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm ridge regression matrices
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive parameters
        self.base_forgetting = 0.98  # Base forgetting factor
        self.forgetting = self.base_forgetting
        self.base_alpha = 1.0        # Base exploration parameter
        self.alpha = self.base_alpha
        # Error tracking
        self.recent_errors = []
        self.error_window = 50
        # Time tracking for deadline safety
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill
        # Track best arm per context pattern for fallback
        self.arm_rewards = {a: [] for a in range(n_arms)}

    def _compute_adaptive_params(self):
        """Compute adaptive forgetting and alpha based on recent prediction errors."""
        if len(self.recent_errors) >= 10:
            recent = self.recent_errors[-self.error_window:]
            mean_err = np.mean(np.abs(recent))
            # Scale forgetting: higher error -> more forgetting (closer to 0)
            # Clamp between 0.90 and 0.999
            self.forgetting = np.clip(0.999 - 0.05 * mean_err, 0.90, 0.999)
            # Scale alpha: higher error -> more exploration
            self.alpha = np.clip(self.base_alpha + 0.5 * mean_err, 0.5, 3.0)
        else:
            self.forgetting = self.base_forgetting
            self.alpha = self.base_alpha

    def _is_deadline_near(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: if we're running low on time, use a simple heuristic
        if self._is_deadline_near():
            # Pick the arm with highest average reward so far
            if any(len(r) > 0 for r in self.arm_rewards.values()):
                avg_rewards = [np.mean(r) if r else 0.0 for r in self.arm_rewards.values()]
                return int(np.argmax(avg_rewards))
            return 0

        self._compute_adaptive_params()

        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mean_reward = theta @ context
                variance = context @ inv @ context
                score = mean_reward + self.alpha * np.sqrt(max(variance, 0))
            except np.linalg.LinAlgError:
                score = 0.0
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction error for adaptive params
        try:
            inv = np.linalg.inv(self.A[arm])
            theta = inv @ self.b[arm]
            predicted = theta @ context
            error = abs(reward - predicted)
        except np.linalg.LinAlgError:
            error = abs(reward)

        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        # Update the arm's model with forgetting
        self.A[arm] = self.forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forgetting * self.b[arm] + reward * context

        # Track rewards for fallback
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > 100:
            self.arm_rewards[arm].pop(0)
# EVOLVE-BLOCK-END
