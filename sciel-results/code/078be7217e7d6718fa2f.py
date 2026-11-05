# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Context-free EMA bandit with aggressive forgetting for non-stationary drift."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm exponential moving averages of rewards
        self.ema = [0.0] * n_arms
        # Per-arm visit counts
        self.visits = [0] * n_arms
        # Aggressive forgetting factor
        self.gamma = 0.90
        # Exploration parameter (UCB-style)
        self.alpha = 1.0
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0
        self.round_count = 0
        # Track best arm so far for safety
        self.best_arm = 0
        self.best_score = -np.inf
        # Per-arm reward history for variance estimation
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history = 50

    def select(self, context):
        self.round_count += 1
        if time.time() - self.start_time > self.deadline:
            return self.best_arm

        if self.round_count <= self.n:
            # Initial exploration: try each arm at least once
            return (self.round_count - 1) % self.n

        # Compute UCB scores using EMA and visit counts
        scores = np.zeros(self.n)
        for a in range(self.n):
            if self.visits[a] > 0:
                # UCB: mean + alpha * sqrt(2 * log(total_visits) / visits[a])
                total_visits = sum(self.visits)
                log_term = np.log(max(total_visits, 1))
                bonus = self.alpha * np.sqrt(2.0 * log_term / self.visits[a])
                scores[a] = self.ema[a] + bonus
            else:
                scores[a] = np.inf  # Force exploration of unvisited arms

        best_idx = int(np.argmax(scores))
        self.best_arm = best_idx
        self.best_score = scores[best_idx]

        return best_idx

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update EMA with aggressive forgetting
        self.ema[arm] = self.gamma * self.ema[arm] + (1.0 - self.gamma) * reward

        # Track reward history for potential variance estimation
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history:
            self.reward_history[arm].pop(0)

        self.visits[arm] += 1

        # Adapt alpha: reduce exploration over time but not too aggressively
        self.alpha = max(0.5, self.alpha * 0.9999)
# EVOLVE-BLOCK-END
