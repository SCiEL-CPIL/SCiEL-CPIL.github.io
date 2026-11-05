# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm exponentially weighted linear regression (forgetting factor)
    - UCB exploration with adaptive beta
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Forgetting factor for non-stationarity
        self.gamma = 0.99
        # Ridge regularization
        self.ridge = 1e-3
        # Base exploration coefficient
        self.beta_base = 1.0
        # Adaptive exploration scale
        self.beta_scale = 1.0

        # Per-arm state: A (dim x dim), b (dim,)
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Track rounds for adaptive beta
        self.round = 0
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 140s safety margin (kill at 150s)

        # Cache inverse for speedup
        self.A_inv = [np.eye(dim) / self.ridge for _ in range(n_arms)]

    def _update_inverse(self, arm):
        """Update the inverse of A[arm] using Sherman-Morrison or direct inversion."""
        # For small dim, direct inversion is fine
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.ridge

    def select(self, context):
        """Select arm using UCB scores."""
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: pick arm 0
            return 0

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mean = theta @ context
            # UCB bonus
            variance = context @ self.A_inv[a] @ context
            scores[a] = mean + self.beta_base * self.beta_scale * np.sqrt(max(variance, 0.0))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model for the given arm with observed reward."""
        # Deadline check
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Exponentially weighted update
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Update inverse
        self._update_inverse(arm)

        # Adaptive beta: reduce exploration over time, but keep some for non-stationarity
        self.round += 1
        # Slowly decay exploration, but maintain a floor
        self.beta_scale = max(0.5, 1.0 / np.sqrt(1 + 0.01 * self.round))

# EVOLVE-BLOCK-END
