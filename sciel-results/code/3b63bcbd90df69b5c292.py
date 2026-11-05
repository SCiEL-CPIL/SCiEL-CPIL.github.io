"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

class Policy:
    """Sliding Window Linear Bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 30  # Number of recent samples to keep per arm
        self.beta = 2.0        # Exploration parameter
        self.lambda_reg = 0.1  # Ridge regularization
        # Store recent observations (context, reward) for each arm
        self.windows = [deque(maxlen=self.window_size) for _ in range(self.n)]
        self.t = 0

    def _get_params(self, a):
        """Compute A_inv, theta, and variance term for arm a based on its window."""
        if len(self.windows[a]) == 0:
            # No data yet: use prior
            A = self.lambda_reg * np.eye(self.d)
            b = np.zeros(self.d)
        else:
            A = self.lambda_reg * np.eye(self.d)
            b = np.zeros(self.d)
            for ctx, rew in self.windows[a]:
                A += np.outer(ctx, ctx)
                b += rew * ctx
        
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d) / self.lambda_reg
        
        theta = A_inv @ b
        return A_inv, theta

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            A_inv, theta = self._get_params(a)
            pred = theta @ context
            var = context @ A_inv @ context
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        self.windows[arm].append((context, reward))
        self.t += 1
# EVOLVE-BLOCK-END
