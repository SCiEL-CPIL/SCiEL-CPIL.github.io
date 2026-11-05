"""Non-stationary linear contextual bandit with multi-scale ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Multi-scale ensemble LinUCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        
        # Multiple scales with geometrically spaced forgetting factors
        self.scales = [0.99, 0.95, 0.85, 0.70]  # short to long memory
        self.n_scales = len(self.scales)
        
        # Per-arm, per-scale precision and cross-moment
        self.A = [[self.I.copy() for _ in range(self.n_scales)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]
        
        # Track recent residuals per arm per scale to select best scale
        self.residual_history = [[[] for _ in range(self.n_scales)] for _ in range(n_arms)]
        self.max_history = 20
        
        # UCB exploration parameter
        self.alpha = 1.0
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill
        
        # Initialize best scale for each arm
        self.best_scale = [0] * n_arms

    def _cholesky_inv(self, M):
        """Stable inverse via Cholesky, with fallback to pinv."""
        try:
            L = np.linalg.cholesky(M)
            inv_L = np.linalg.inv(L)
            inv_M = inv_L.T @ inv_L
            return inv_M
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def _get_best_scale(self, arm):
        """Select scale with lowest recent residual variance."""
        if not self.residual_history[arm]:
            return 0  # Default to longest memory
        
        # Compute variance of residuals for each scale's predictions
        # We'll use the most recent residuals as a proxy
        recent = self.residual_history[arm][-self.max_history:]
        
        # For simplicity, we'll track which scale performed best
        # by maintaining a weighted score based on recent performance
        return self.best_scale[arm]

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return best arm based on current predictions
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                scale = self.best_scale[a]
                A_inv = self._cholesky_inv(self.A[a][scale])
                theta = A_inv @ self.b[a][scale]
                mu = theta @ context
                score = mu
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm
        
        scores = []
        for a in range(self.n):
            # Use the best-performing scale for this arm
            scale = self.best_scale[a]
            A_inv = self._cholesky_inv(self.A[a][scale])
            theta = A_inv @ self.b[a][scale]
            mu = theta @ context
            # Uncertainty: sqrt(context^T A^{-1} context)
            var = context @ A_inv @ context
            std = np.sqrt(max(var, 1e-12))
            # UCB score
            score = mu + self.alpha * std
            scores.append(score)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Update all scales for this arm
        for s in range(self.n_scales):
            decay = self.scales[s]
            # Apply forgetting
            self.A[arm][s] = decay * self.A[arm][s]
            self.b[arm][s] = decay * self.b[arm][s]
            # Add new observation
            self.A[arm][s] = self.A[arm][s] + np.outer(context, context)
            self.b[arm][s] = self.b[arm][s] + reward * context
        
        # Compute residuals for all scales and update history
        for s in range(self.n_scales):
            A_inv_s = self._cholesky_inv(self.A[arm][s])
            theta_s = A_inv_s @ self.b[arm][s]
            mu_s = theta_s @ context
            residual_s = (reward - mu_s) ** 2
            self.residual_history[arm][s].append(residual_s)
            if len(self.residual_history[arm][s]) > self.max_history:
                self.residual_history[arm][s] = self.residual_history[arm][s][-self.max_history:]
        
        # Select best scale based on recent performance
        # For each scale, compute average squared residual over recent window
        best_scale_idx = 0
        best_score = np.inf
        
        for s in range(self.n_scales):
            if self.residual_history[arm][s]:
                avg_residual = np.mean(self.residual_history[arm][s])
            else:
                avg_residual = np.inf
            if avg_residual < best_score:
                best_score = avg_residual
                best_scale_idx = s
        
        self.best_scale[arm] = best_scale_idx
# EVOLVE-BLOCK-END
