"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Sliding Window Ridge Regression for Non-Stationary Linear Bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.W = 50  # Sliding window size
        self.alpha = 1.0  # Exploration parameter
        self.ridge = 1.0  # Ridge regularization
        self.buffers = [[] for _ in range(n_arms)]
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin before 150s kill

    def _update_stats(self, arm, context, reward):
        # Add new sample
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.buffers[arm].append((context, reward))

        # Remove oldest sample if window exceeded
        if len(self.buffers[arm]) > self.W:
            old_ctx, old_rwd = self.buffers[arm].pop(0)
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_rwd * old_ctx

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: Greedy based on current estimates
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(float(theta @ context))
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            try:
                # Solve A * theta = b
                theta = np.linalg.solve(self.A[a], self.b[a])
                pred = float(theta @ context)

                # Solve A * v = context for uncertainty
                v = np.linalg.solve(self.A[a], context)
                unc = np.sqrt(float(context @ v))

                scores.append(pred + self.alpha * unc)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._update_stats(arm, context, reward)
# EVOLVE-BLOCK-END
