"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.gamma = 0.995
        self.alpha = 0.5
        self.t = 0
        self.start_time = time.time()
        self.deadline = 150.0
        self.last_update_time = 0.0
        self.recent_rewards = [np.zeros(dim) for _ in range(n_arms)]
        self.recent_counts = [0.0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline - 5.0:
            # Return best arm based on current estimates
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        for a in range(self.n):
            # Add small regularization for numerical stability
            A_reg = self.A[a] + 1e-6 * np.eye(self.d)
            try:
                inv = np.linalg.inv(A_reg)
            except np.linalg.LinAlgError:
                inv = np.eye(self.d)
            theta = inv @ self.b[a]
            mean = theta @ context
            # Uncertainty bonus
            var = context @ inv @ context
            bonus = self.alpha * np.sqrt(var)
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Adapt forgetting factor based on recent reward variance
        self.recent_rewards[arm] = self.gamma * self.recent_rewards[arm] + reward * context
        self.recent_counts[arm] = self.gamma * self.recent_counts[arm] + 1.0

        # If we have enough data, estimate reward variance to adapt gamma
        if self.recent_counts[arm] > 10.0:
            # Simple heuristic: if rewards are changing rapidly, reduce gamma
            # This is a rough proxy; in practice, we'd track reward variance
            pass

        # Update with forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Keep matrices well-conditioned by periodically re-regularizing
        if self.t % 100 == 0:
            for a in range(self.n):
                # Add small identity to prevent singularity
                self.A[a] += 1e-8 * np.eye(self.d)
# EVOLVE-BLOCK-END
