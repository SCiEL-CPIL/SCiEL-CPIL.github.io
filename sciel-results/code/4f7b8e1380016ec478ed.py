# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """LinUCB with Change-Point Detection and Reset."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Initialize A^{-1} and b for each arm
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Exploration parameter
        self.C = 1.0
        self.round_count = 0
        
        # Change detection parameters
        self.window_size = 10
        self.error_history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.reset_threshold_factor = 2.0  # Reset if error > mean + factor * std
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 120.0

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline - 5.0:
            # Fallback: greedy based on current means
            scores = [ (self.A_inv[a] @ self.b[a]) @ context for a in range(self.n) ]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            # Standard LinUCB score
            scores.append(theta @ context + self.C * np.sqrt(context @ inv @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        
        # Deadline guard
        if time.time() - self._start_time > self._deadline - 2.0:
            return

        # 1. Compute prediction and error
        inv = self.A_inv[arm]
        theta = inv @ self.b[arm]
        pred = theta @ context
        error = reward - pred
        
        # 2. Change Detection
        hist = self.error_history[arm]
        if len(hist) >= 5:
            mean_err = np.mean(hist)
            std_err = np.std(hist)
            # If the current error is an outlier (large deviation), reset
            if abs(error) > mean_err + self.reset_threshold_factor * std_err + 1e-6:
                # Reset the model for this arm
                self.A_inv[arm] = np.eye(self.d)
                self.b[arm] = np.zeros(self.d)
                # Clear history to start fresh estimation
                hist.clear()

        # 3. Update history
        hist.append(error)

        # 4. Standard LinUCB Update
        # Update b
        self.b[arm] += reward * context
        
        # Update A^{-1} using Sherman-Morrison
        # A_new = A_old + x x^T
        # V_new = V_old - V_old x x^T V_old / (1 + x^T V_old x)
        V = self.A_inv[arm]
        x = context
        Vx = V @ x
        denom = 1.0 + x @ Vx
        self.A_inv[arm] = V - np.outer(Vx, Vx) / denom
        
        # Symmetrize
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
# EVOLVE-BLOCK-END
