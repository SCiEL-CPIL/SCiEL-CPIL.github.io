"""Thompson Sampling with sliding window and Cholesky updates for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """Thompson Sampling with sliding window for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.window_size = 80
        self.reg = 1.0  # Ridge regularization
        
        # Per-arm storage for recent (context, reward) pairs
        self.histories = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Per-arm Cholesky factor of (X^T X + reg * I)
        self.cholesky = [np.eye(dim) for _ in range(n_arms)]
        # Per-arm X^T y vector
        self.Xty = [np.zeros(dim) for _ in range(n_arms)]
        # Per-arm X^T X matrix
        self.XtX = [self.reg * np.eye(dim) for _ in range(n_arms)]
        
        # Track whether Cholesky is valid
        self.chol_valid = [True] * n_arms
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 110.0  # Leave 10s safety margin from 120s limit

    def _recompute_cholesky(self, arm):
        """Recompute Cholesky factor for arm."""
        try:
            self.cholesky[arm] = np.linalg.cholesky(self.XtX[arm])
            self.chol_valid[arm] = True
        except np.linalg.LinAlgError:
            # Fallback: add small jitter
            jitter = 1e-6 * np.eye(self.dim)
            self.cholesky[arm] = np.linalg.cholesky(self.XtX[arm] + jitter)
            self.chol_valid[arm] = True

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        samples = []
        
        for a in range(self.n_arms):
            if not self.chol_valid[a]:
                self._recompute_cholesky(a)
            
            L = self.cholesky[a]
            
            # Solve L * theta = Xty using triangular solve
            try:
                theta_hat = np.linalg.solve(L, self.Xty[a])
            except np.linalg.LinAlgError:
                self._recompute_cholesky(a)
                L = self.cholesky[a]
                theta_hat = np.linalg.solve(L, self.Xty[a])
            
            # Compute covariance: (X^T X + reg I)^-1 = L^-1 * L^-T
            # For sampling: theta ~ N(theta_hat, (X^T X + reg I)^-1)
            # We can sample by: theta = theta_hat + L^-T * z where z ~ N(0, I)
            # Actually: Cov = (L L^T)^-1 = L^-T L^-1
            # So: theta = theta_hat + L^-T * z
            
            # Solve L^T * v = z, then theta = theta_hat + v
            z = np.random.randn(self.dim)
            try:
                v = np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                v = z  # fallback
            
            theta = theta_hat + v
            samples.append(theta @ context)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # Add to history
        self.histories[arm].append((context.copy(), reward))
        
        # Recompute XtX and Xty from the window
        if len(self.histories[arm]) == 1:
            self.XtX[arm] = np.outer(context, context) + self.reg * np.eye(self.dim)
            self.Xty[arm] = reward * context
        else:
            self.XtX[arm] = self.XtX[arm] + np.outer(context, context)
            self.Xty[arm] = self.Xty[arm] + reward * context
        
        # When window is full, we need to remove the oldest observation
        # But deque with maxlen automatically drops the oldest
        # So we need to recompute when the window wraps around
        # For simplicity, let's recompute from scratch when window is full
        
        if len(self.histories[arm]) >= self.window_size:
            # Recompute from the full window
            XtX = self.reg * np.eye(self.dim)
            Xty = np.zeros(self.dim)
            for ctx, rew in self.histories[arm]:
                XtX += np.outer(ctx, ctx)
                Xty += rew * ctx
            self.XtX[arm] = XtX
            self.Xty[arm] = Xty
        
        # Update Cholesky
        self._recompute_cholesky(arm)
# EVOLVE-BLOCK-END
