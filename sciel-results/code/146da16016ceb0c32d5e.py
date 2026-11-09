import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.start_time = time.time()
        
        # Hyperparameters
        # Adaptive gamma: Start high (slow forgetting) and decay to lower (fast forgetting)
        self.gamma_base = 0.995
        self.gamma_min = 0.97
        self.gamma_decay = 500
        
        # Exploration parameters
        self.beta_ucb = 1.5
        self.ts_weight = 0.8
        
        # Drift detection parameters
        self.reset_threshold = 1.5
        self.ewma_alpha = 0.15
        
        # Initialize per-arm models
        # A: Inverse covariance (regularized)
        # b: Linear system vector
        # L: Cholesky factor of A for solving
        eye = np.eye(dim)
        self.A = [eye.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Pre-compute Cholesky for initial identity
        reg_eye = eye + 1e-6 * np.eye(dim)
        self.L = [np.linalg.cholesky(reg_eye) for _ in range(n_arms)]
        
        # Residual tracking for drift detection
        self.residuals = [0.0 for _ in range(n_arms)]
        self.t = 0
        self._rng = np.random.RandomState(42)
        
        # Safety margin
        self._SAFE = 140.0

    def _update_cholesky(self, a):
        """Recompute Cholesky factor for arm a."""
        A = self.A[a]
        # Add small regularization for numerical stability
        M = A + 1e-6 * np.eye(self.dim)
        try:
            self.L[a] = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # If it fails, try larger regularization
            M = A + 1e-3 * np.eye(self.dim)
            try:
                self.L[a] = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                # Last resort: reset to identity
                self.A[a] = np.eye(self.dim).copy()
                self.b[a] = np.zeros(self.dim)
                self.L[a] = np.linalg.cholesky(np.eye(self.dim) + 1e-6 * np.eye(self.dim))
                self.residuals[a] = 0.0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        # Deadline guard
        if time.time() - self.start_time > self._SAFE:
            # Fallback to greedy
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n_arms):
                try:
                    y = np.linalg.solve(self.L[a], self.b[a])
                    theta = np.linalg.solve(self.L[a].T, y)
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
            L = self.L[a]
            b = self.b[a]
            
            # Solve for theta: theta = A^{-1} b
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
            noise = self._rng.randn() * self.ts_weight * std
            score = mean + self.beta_ucb * std + noise
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        a = int(arm)
        r = float(reward)
        
        # Adaptive gamma: decrease over time to adapt faster to non-stationarity
        progress = min(1.0, self.t / self.gamma_decay)
        g = self.gamma_base - (self.gamma_base - self.gamma_min) * progress
        
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
            prev_res = self.residuals[a]
            new_res = (1 - self.ewma_alpha) * prev_res + self.ewma_alpha * resid_sq
            self.residuals[a] = new_res
            
            # If residual variance is high, reset the model for this arm
            if new_res > self.reset_threshold:
                eye = np.eye(self.dim)
                self.A[a] = eye.copy()
                self.b[a] = np.zeros(self.dim)
                self.L[a] = np.linalg.cholesky(eye + 1e-6 * np.eye(self.dim))
                self.residuals[a] = 0.0
                
        self.t += 1
