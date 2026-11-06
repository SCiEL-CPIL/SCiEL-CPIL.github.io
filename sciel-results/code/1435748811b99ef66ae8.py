import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.ridge = 1.0
        
        # Sliding window size
        self.window_size = 50
        
        # Per-arm state
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # History for sliding window
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Thompson Sampling noise
        self.noise_var = 1.0
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0
        
        # Random state for reproducible sampling (if needed, but np.random is fine)
        self.rng = np.random.default_rng()

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _smw_add(self, arm, x):
        """Add x x^T to A[arm] and update inverse."""
        A_inv = self.inv_A[arm]
        tmp = A_inv @ x
        denom = 1.0 + x @ tmp
        if abs(denom) < 1e-10:
            # Fallback
            self.A[arm] = self.A[arm] + np.outer(x, x)
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        else:
            self.inv_A[arm] = A_inv - np.outer(tmp, tmp) / denom
            self.A[arm] = self.A[arm] + np.outer(x, x)
        self.b[arm] = self.b[arm] + 0 # b is updated separately with reward
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def _smw_subtract(self, arm, x):
        """Subtract x x^T from A[arm] and update inverse."""
        A_inv = self.inv_A[arm]
        tmp = A_inv @ x
        denom = 1.0 - x @ tmp
        if abs(denom) < 1e-10 or denom <= 0:
            # If subtraction makes matrix singular or indefinite, recompute from scratch
            # This is a rare edge case, but safer to recompute if window is large
            # For now, fallback to full inverse of current A
            self.A[arm] = self.A[arm] - np.outer(x, x)
            try:
                self.inv_A[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.inv_A[arm] = np.eye(self.d)
        else:
            self.inv_A[arm] = A_inv + np.outer(tmp, tmp) / denom
            self.A[arm] = self.A[arm] - np.outer(x, x)
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick best mean
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Thompson Sampling: Sample theta from posterior
            # theta ~ N(theta_hat, noise_var * inv_A)
            # We sample z ~ N(0, I), then theta_sample = theta_hat + noise_var * sqrt(inv_A) @ z
            # Since inv_A is symmetric PSD, we can use Cholesky or just eigen-decomp if small
            # For small dim, direct sampling using Cholesky of inv_A is efficient
            try:
                L = np.linalg.cholesky(self.inv_A[a])
                z = self.rng.normal(size=self.d)
                theta_sample = self.theta[a] + self.noise_var * (L @ z)
            except np.linalg.LinAlgError:
                # Fallback to mean if Cholesky fails (shouldn't happen with ridge)
                theta_sample = self.theta[a]
            
            scores.append(context @ theta_sample)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        # Check if we need to evict
        if len(self.history[arm]) == self.window_size:
            old_x, old_r = self.history[arm][0]
            # Subtract old contribution
            self._smw_subtract(arm, old_x)
            self.b[arm] = self.b[arm] - old_r * old_x
            
        # Add new contribution
        self.history[arm].append((context, reward))
        self._smw_add(arm, context)
        self.b[arm] = self.b[arm] + reward * context
        
        # Update theta after b change
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]
