# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting for non-stationary adaptation."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.base_gamma = 0.90  # More aggressive decay to adapt faster to non-stationarity
        self.alpha = 1.5        # Higher exploration bonus for better arm coverage
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = 110.0  # Safety margin before 120s limit

    def _check_deadline(self):
        """Return True if we should stop exploring to avoid timeout."""
        return (time.time() - self._start_time) > self._deadline

    def _adaptive_gamma(self):
        """Adapt forgetting rate based on time elapsed."""
        elapsed = time.time() - self._start_time
        # Start with aggressive forgetting, gradually increase as we learn
        # This helps with initial exploration and later exploitation
        if elapsed < 5.0:
            return 0.85  # Very fast forgetting initially
        elif elapsed < 15.0:
            return 0.90
        else:
            return 0.93  # Slower forgetting once patterns are learned

    def select(self, context):
        # Check deadline - if near timeout, exploit best known arm
        if self._check_deadline():
            # Exploit: pick arm with highest mean reward estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    inv = np.linalg.inv(self.A[a])
                    theta = inv @ self.b[a]
                    pred = theta @ context
                    if pred > best_score:
                        best_score = pred
                        best_arm = a
                except np.linalg.LinAlgError:
                    continue
            return best_arm

        scores = []
        gamma = self._adaptive_gamma()
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                pred = theta @ context
                # Calculate UCB bonus
                var = context @ inv @ context
                bonus = self.alpha * np.sqrt(max(var, 0))
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                scores.append(np.random.randn())
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Use adaptive gamma for decay
        gamma = self._adaptive_gamma()
        
        # Decay existing statistics
        self.A[arm] = gamma * self.A[arm]
        self.b[arm] = gamma * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
# EVOLVE-BLOCK-END
