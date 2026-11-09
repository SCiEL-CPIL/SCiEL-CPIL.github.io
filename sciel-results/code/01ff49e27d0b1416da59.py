"""
Non-stationary linear contextual bandit policy.
Combines Discounted UCB (fast, robust) with Thompson Sampling's exploration
and adaptive drift detection (reset on high residual variance).
Uses Cholesky decomposition for numerical stability and speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_SAFE = 140.0

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.start_time = time.time()
        
        # Hyperparameters
        # gamma: Forgetting factor. 0.99 is a good balance.
        # beta: Exploration bonus coefficient for UCB part.
        # ts_weight: Weight for Thompson Sampling noise.
        # reset_threshold: Threshold for residual variance to trigger reset.
        self.gamma = 0.99
        self.beta = 2.0
        self.ts_weight = 1.0
        self.reset_threshold = 2.0
        self.ewma_alpha = 0.1  # Smoothing factor for residual variance
        
        # Initialize per-arm models
        # A: Inverse covariance (regularized)
        # b: Linear system vector
        # L: Cholesky factor of A (or A+I) for sampling/solving
        eye = np.eye(dim)
        self.A = [eye.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.L = [np.linalg.cholesky(eye + 1e-6 * np.eye(dim)) for _ in range(n_arms)]
        
        # Residual tracking for drift detection
        self.residuals = [np.zeros(1) for _ in range(n_arms)] # EWMA of squared residuals
        self.t = 0

    def _update_cholesky(self, a):
        """Recompute Cholesky factor for arm a. d is small, so this is fast."""
        A = self.A[a]
        # Add small regularization for numerical stability
        M = A + 1e-6 * np.eye(self.dim)
        try:
            self.L[a] = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # If it fails, try larger regularization
            self.L[a] = np.linalg.cholesky(M + 1e-3 * np.eye(self.dim))

    def select(self, context):
        """
        Selects an arm using a hybrid UCB/Thompson Sampling rule.
        """
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        # Deadline guard
        if time.time() - self.start_time > _SAFE:
            # Fallback to greedy
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n_arms):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    val = float(np.dot(theta, context))
                    if val > best_val:
                        best_val = val
                        best_arm = a
                except:
                    pass
            return int(best_arm)

        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            A = self.A[a]
            b = self.b[a]
            L = self.L[a]
            
            # Solve for theta: theta = A^{-1} b
            # Using Cholesky: A = L L^T, so theta = L^{-T} L^{-1} b
            try:
                y = np.linalg.solve(L, b)
                theta = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.dim)
            
            # Predicted mean
            mean = float(np.dot(theta, context))
            
            # Confidence bound: beta * sqrt(x^T A^{-1} x)
            # x^T A^{-1} x = x^T L^{-T} L^{-1} x = ||L^{-1} x||^2
            try:
                v = np.linalg.solve(L.T, context) # L^{-T} x
                var = float(np.dot(v, v))
            except np.linalg.LinAlgError:
                var = 0.0
                
            if var < 0:
                var = 0.0
            
            std = np.sqrt(var)
            
            # Hybrid score: UCB + Thompson noise
            # Thompson noise: sample from N(mean, var)
            # We can approximate by adding noise scaled by std
            noise = np.random.randn() * self.ts_weight * std
            
            score = mean + self.beta * std + noise
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        """
        Updates the linear model for the chosen arm using discounted update.
        """
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        a = int(arm)
        r = float(reward)
        g = self.gamma
        
        # Update A and b
        self.A[a] = g * self.A[a] + np.outer(context, context)
        self.b[a] = g * self.b[a] + r * context
        
        # Update Cholesky factor
        self._update_cholesky(a)
        
        # Drift detection: Update residual EWMA and check for reset
        if self.t > 10: # Warmup
            # Predicted reward
            try:
                y = np.linalg.solve(self.L[a], self.b[a])
                theta = np.linalg.solve(self.L[a].T, y)
                pred = float(np.dot(theta, context))
            except:
                pred = 0.0
                
            resid = r - pred
            resid_sq = resid * resid
            
            # Update EWMA of squared residuals
            prev_res = self.residuals[a][0]
            new_res = (1 - self.ewma_alpha) * prev_res + self.ewma_alpha * resid_sq
            self.residuals[a] = np.array([new_res])
            
            # If residual variance is high, reset the model for this arm
            if new_res > self.reset_threshold:
                eye = np.eye(self.dim)
                self.A[a] = eye.copy()
                self.b[a] = np.zeros(self.dim)
                self.L[a] = np.linalg.cholesky(eye + 1e-6 * np.eye(self.dim))
                self.residuals[a] = np.zeros(1)
                
        self.t += 1

# EVOLVE-BLOCK-END
