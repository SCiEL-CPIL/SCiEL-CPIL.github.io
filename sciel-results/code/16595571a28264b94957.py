"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Non-stationary linear bandit with adaptive forgetting and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0

        # Per-arm state
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]

        # Adaptive forgetting factor
        self.gamma = 0.97
        self.gamma_min = 0.90
        self.gamma_max = 0.998

        # Exploration
        self.alpha = 1.5
        self.alpha_decay = 0.999

        # Reward tracking
        self.reward_history = [deque(maxlen=30) for _ in range(n_arms)]

        # Reset mechanism
        self.ema_reward = 0.0
        self.ema_alpha = 0.1
        self.reset_threshold = -0.5
        self.reset_cooldown = 0

    def select(self, context):
        self.t += 1
        ctx = np.asarray(context, dtype=np.float64)

        # Check deadline
        if time.time() > self.deadline:
            means = [np.mean(list(rh)) if len(rh) > 0 else 0.0 for rh in self.reward_history]
            return int(np.argmax(means))

        scores = []
        for a in range(self.n):
            # Compute theta and uncertainty
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
                pred = theta @ ctx
                uncertainty = np.sqrt(max(ctx @ inv_A @ ctx, 0.0))
            except np.linalg.LinAlgError:
                pred = 0.0
                uncertainty = 1.0

            # Adaptive alpha: scale by sqrt of count to encourage exploration early
            count_scale = 1.0 + 1.0 / (1.0 + self.counts[a] * 0.1)
            score = pred + self.alpha * count_scale * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        self.t += 1

        if time.time() > self.deadline:
            return

        # Update per-arm state with forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = self.gamma * self.b[arm] + reward * ctx
        self.counts[arm] += 1

        # Track reward history
        self.reward_history[arm].append(reward)

        # Adaptive forgetting factor
        if len(self.reward_history[arm]) >= 10:
            recent = list(self.reward_history[arm])[-10:]
            var = np.var(recent)
            mean_r = np.mean(recent)
            if abs(mean_r) > 0.1:
                stability = 1.0 / (1.0 + var / (mean_r ** 2 + 1e-6))
                target_gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * stability
            else:
                target_gamma = (self.gamma_min + self.gamma_max) / 2.0
            self.gamma = 0.95 * self.gamma + 0.05 * target_gamma

        # EMA of reward
        self.ema_reward = (1 - self.ema_alpha) * self.ema_reward + self.ema_alpha * reward

        # Reset mechanism
        if self.ema_reward < self.reset_threshold and self.reset_cooldown <= 0:
            for a in range(self.n):
                self.A[a] = 0.5 * self.A[a] + 0.5 * np.eye(self.d) * 1.0
                self.b[a] = 0.5 * self.b[a]
                self.reward_history[a].clear()
            self.ema_reward = 0.0
            self.reset_cooldown = 50
        elif self.reset_cooldown > 0:
            self.reset_cooldown -= 1
# EVOLVE-BLOCK-END
