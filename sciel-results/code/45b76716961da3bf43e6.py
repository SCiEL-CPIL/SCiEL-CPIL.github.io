# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear contextual bandit using a sliding-window / forgetting
    factor approach with per-arm linear models and UCB exploration.
    
    To handle non-stationarity (changing arm weights), we use:
    1. A forgetting factor applied to the sufficient statistics (A, b) for each arm.
    2. A more aggressive exploration term that adapts to the uncertainty.
    3. Per-arm models rather than a single global model, which is better when
       arms have distinct reward functions.
    
    Key improvements:
    - Use a slightly higher forgetting factor for stability but still responsive.
    - Add a small ridge regularization to prevent singular matrices.
    - Use a more principled UCB with a time-adaptive exploration parameter.
    - Keep per-arm statistics to better track each arm's changing weights.
    - Use solve instead of explicit inverse for numerical stability.
    - Adapt exploration based on recent reward variance per arm.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm sufficient statistics with ridge regularization
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor: balance between tracking non-stationarity and stability
        self.forgetting_factor = 0.98
        # Exploration parameter, can be adapted over time
        self.alpha = 1.5
        self.t = 0
        # Track per-arm visit counts for adaptive exploration
        self.visits = [0] * n_arms
        # Track recent rewards per arm for variance-based exploration
        self.reward_sum = [0.0] * n_arms
        self.reward_sq_sum = [0.0] * n_arms

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Solve for theta = A^{-1} b using solve for numerical stability
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(self.A[a]) @ self.b[a]
            
            # Mean reward estimate
            mean = theta @ context
            
            # Uncertainty term: alpha * sqrt(context^T A^{-1} context)
            # Compute context^T A^{-1} context efficiently
            try:
                inv_A_context = np.linalg.solve(self.A[a], context)
                variance = context @ inv_A_context
            except np.linalg.LinAlgError:
                inv_A = np.linalg.pinv(self.A[a])
                variance = context @ inv_A @ context
            
            # Adaptive alpha based on visit count and reward variance
            base_alpha = self.alpha
            # Increase exploration for arms with high recent variance
            if self.visits[a] > 2:
                mean_r = self.reward_sum[a] / self.visits[a]
                var_r = max(0.0, self.reward_sq_sum[a] / self.visits[a] - mean_r * mean_r)
                variance_bonus = 0.3 * np.sqrt(var_r)
            else:
                variance_bonus = 0.0
            
            visit_factor = 1.0 + 0.5 / (1.0 + self.visits[a] * 0.1)
            adaptive_alpha = base_alpha * visit_factor + variance_bonus
            uncertainty = adaptive_alpha * np.sqrt(max(0.0, variance))
            
            scores.append(mean + uncertainty)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Apply forgetting factor to sufficient statistics
        ff = self.forgetting_factor
        self.A[arm] = ff * self.A[arm] + np.outer(context, context)
        self.b[arm] = ff * self.b[arm] + reward * context
        
        # Track visits
        self.visits[arm] += 1
        
        # Update reward statistics with forgetting
        self.reward_sum[arm] = ff * self.reward_sum[arm] + reward
        self.reward_sq_sum[arm] = ff * self.reward_sq_sum[arm] + reward * reward
        
        # Gradually reduce base exploration as we learn more overall
        # But keep a floor to maintain some exploration for non-stationarity
        self.alpha = max(0.8, 1.5 / (1.0 + self.t * 0.002))
# EVOLVE-BLOCK-END
