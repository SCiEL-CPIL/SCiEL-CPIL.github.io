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
    
    Key improvements over the seed:
    - Use a slightly higher forgetting factor for stability but still responsive.
    - Add a small ridge regularization to prevent singular matrices.
    - Use a more principled UCB with a time-adaptive exploration parameter.
    - Keep per-arm statistics to better track each arm's changing weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm sufficient statistics with ridge regularization
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor: balance between tracking non-stationarity and stability
        self.forgetting_factor = 0.99
        # Exploration parameter, can be adapted over time
        self.alpha = 1.0
        self.t = 0
        # Track per-arm visit counts for adaptive exploration
        self.visits = [0] * n_arms

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            try:
                inv_A = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                inv_A = np.linalg.pinv(self.A[a])
            theta = inv_A @ self.b[a]
            
            # Mean reward estimate
            mean = theta @ context
            
            # Uncertainty term: alpha * sqrt(context^T A^{-1} context)
            # Use a slightly adaptive alpha based on visit count
            adaptive_alpha = self.alpha * (1.0 + 0.5 / (1.0 + self.visits[a] * 0.1))
            uncertainty = adaptive_alpha * np.sqrt(max(0.0, context @ inv_A @ context))
            
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
        
        # Gradually reduce exploration as we learn more overall
        # But keep a floor to maintain some exploration for non-stationarity
        self.alpha = max(0.5, 1.0 / (1.0 + self.t * 0.001))
# EVOLVE-BLOCK-END
