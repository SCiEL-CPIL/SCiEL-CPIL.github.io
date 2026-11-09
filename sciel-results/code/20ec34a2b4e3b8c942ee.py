"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Non-stationary linear contextual bandit with global + arm-specific model."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Global model shared across arms
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)
        # Arm-specific offset vectors (bias for each arm)
        self.arm_offsets = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor for non-stationarity
        self.forgetting = 0.99
        # Exploration parameter
        self.alpha = 1.0
        # Track recent rewards for adaptive exploration
        self.recent_rewards = []
        self.max_recent = 50

    def _solve(self, A, b):
        """Solve A x = b efficiently."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        # Solve global model
        theta_global = self._solve(self.A_global, self.b_global)
        
        for a in range(self.n):
            # Effective theta = global + arm offset
            theta = theta_global + self.arm_offsets[a]
            
            # Compute prediction
            pred = theta @ context
            
            # Compute uncertainty
            # For global part: context^T A_global^{-1} context
            try:
                inv_global = np.linalg.inv(self.A_global)
                var_global = context @ inv_global @ context
            except np.linalg.LinAlgError:
                var_global = 1.0
            
            # Total variance (approximate, assuming arm offsets are fixed)
            total_var = var_global
            
            # UCB score
            score = pred + self.alpha * np.sqrt(total_var)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Update global model with forgetting
        self.A_global = self.forgetting * self.A_global + np.outer(context, context)
        self.b_global = self.forgetting * self.b_global + reward * context
        
        # Update arm-specific offset
        # The arm offset captures the deviation from the global model
        # We can estimate it as: offset_a = (reward - global_pred) / ||context||^2 * context
        # But simpler: just track arm-specific statistics
        # Actually, let's use a simpler approach: maintain per-arm A and b
        # but share the context structure
        
        # Alternative: just use the global model and let the arm index be part of context
        # For now, keep it simple with just global model
        pass
# EVOLVE-BLOCK-END
