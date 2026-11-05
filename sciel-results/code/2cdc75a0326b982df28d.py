"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


class Policy:
    """Sliding Window Linear Bandit with adaptive exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 60
        self.lam = 1.0
        self.hists = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.t = 0
        self.base_alpha = 1.0
        self.alpha_decay = 0.999

    def select(self, context):
        self.t += 1
        alpha = self.base_alpha * (self.alpha_decay ** self.t)
        scores = []
        for a in range(self.n):
            if len(self.hists[a]) < self.d:
                scores.append(np.random.randn())
                continue
            X = np.array([x for x, r in self.hists[a]])
            y = np.array([r for x, r in self.hists[a]])
            # Add regularization
            X_reg = X
            y_reg = y
            try:
                theta, _, _, _ = np.linalg.lstsq(X_reg + self.lam * np.eye(self.d), y_reg, rcond=None)
                pred = theta @ context
                # Compute residual variance for adaptive exploration
                residuals = y_reg - X_reg @ theta
                var = np.var(residuals) if len(residuals) > 1 else 1.0
                scores.append(pred + alpha * np.sqrt(var + 1.0))
            except np.linalg.LinAlgError:
                scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.hists[arm].append((context, reward))
# EVOLVE-BLOCK-END
