"""
Non-stationary linear contextual bandit policy.
Uses a single global linear model with arm-augmented context (one-hot arm + context).
Combines Discounted UCB with adaptive forgetting and drift detection.
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
        
        # Feature dimension: one-hot arm (n_arms) + context (dim)
        self.feature_dim = n_arms + dim
        
        # Hyperparameters
        self.gamma_base = 0.99
        self.gamma_min = 0.95
        self.gamma_decay = 400
        self.beta = 1.2
        self.ewma_alpha = 0.2
        self.reset_threshold = 2.0
        
        # Initialize global model
        eye = np.eye(self.feature_dim)
        self.A = eye.copy()
        self.b = np.zeros(self.feature_dim)
        
        # Cholesky factor for numerical stability
        reg = 1e-6 * np.eye(self.feature_dim)
        self.L = np.linalg.cholesky(self.A + reg)
        
        # Residual tracking
        self.residual_ewma = 0.0
        self.t = 0
        self._rng = np.random.RandomState(42)

    def _augment(self, context, arm):
        """Create augmented feature vector: one-hot arm + context."""
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
        onehot = np.zeros(self.n_arms)
        onehot[int(arm)] = 1.0
        return np.concatenate([onehot, context])

    def _update_cholesky(self):
        """Recompute Cholesky factor."""
        reg = 1e-6 * np.eye(self.feature_dim)
        M = self.A + reg
        try:
            self.L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(M + 1e-3 * np.eye(self.feature_dim))

    def select(self, context):
        """Select arm using UCB with Thompson Sampling noise."""
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        # Deadline guard
        if time.time() - self.start_time > _SAFE:
            # Fallback to greedy
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n_arms):
                x = self._augment(context, a)
                try:
                    y = np.linalg.solve(self.L, self.b)
                    theta = np.linalg.solve(self.L.T, y)
                    val = float(np.dot(theta, x))
                    if val > best_val:
                        best_val = val
                        best_arm = a
                except:
                    pass
            return int(best_arm)

        # Solve for theta once
        try:
            y = np.linalg.solve(self.L, self.b)
            theta = np.linalg.solve(self.L.T, y)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.feature_dim)
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            x = self._augment(context, a)
            
            # Predicted mean
            mean = float(np.dot(theta, x))
            
            # Confidence: x^T A^{-1} x = ||L^{-1} x||^2
            try:
                v = np.linalg.solve(self.L.T, x)
                var = float(np.dot(v, v))
            except np.linalg.LinAlgError:
                var = 0.0
            
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # UCB + small Thompson noise for tie-breaking
            noise = self._rng.randn() * 0.3 * std
            score = mean + self.beta * std + noise
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        """Update the global model with discounted linear regression."""
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        a = int(arm)
        r = float(reward)
        x = self._augment(context, a)
        
        # Adaptive gamma
        progress = min(1.0, self.t / self.gamma_decay)
        g = self.gamma_base - (self.gamma_base - self.gamma_min) * progress
        
        # Discounted update
        self.A = g * self.A + np.outer(x, x)
        self.b = g * self.b + r * x
        
        # Update Cholesky
        self._update_cholesky()
        
        # Drift detection
        if self.t > 20:
            try:
                y = np.linalg.solve(self.L, self.b)
                theta = np.linalg.solve(self.L.T, y)
                pred = float(np.dot(theta, x))
            except:
                pred = 0.0
                
            resid = r - pred
            resid_sq = resid * resid
            self.residual_ewma = (1 - self.ewma_alpha) * self.residual_ewma + self.ewma_alpha * resid_sq
            
            if self.residual_ewma > self.reset_threshold:
                eye = np.eye(self.feature_dim)
                self.A = eye.copy()
                self.b = np.zeros(self.feature_dim)
                reg = 1e-6 * np.eye(self.feature_dim)
                self.L = np.linalg.cholesky(self.A + reg)
                self.residual_ewma = 0.0
                
        self.t += 1

# EVOLVE-BLOCK-END
