import numpy as np
from collections import deque

class Policy:
    """
    Non-stationary Linear Bandit with Adaptive Forgetting Factor.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0
        # Initialize A and b
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.gammas = [0.99 for _ in range(n_arms)]
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.gamma_step = 0.005
        
        # History for adaptation
        self.err_hist = [deque(maxlen=20) for _ in range(n_arms)]
        self.reward_hist = deque(maxlen=100)
        
        # Exploration constant
        self.alpha = 1.5

    def select(self, context):
        scores = []
        for a in range(self.n):
            try:
                # Solve A * theta = b
                theta = np.linalg.solve(self.A[a], self.b[a])
                # Solve A * v = context to get v = A^-1 context
                # Then variance is context^T v
                v = np.linalg.solve(self.A[a], context)
                var = context @ v
                if var < 0:
                    var = 0.0
                ucb = theta @ context + self.alpha * np.sqrt(var)
            except np.linalg.LinAlgError:
                ucb = 0.0
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track reward scale for normalization
        self.reward_hist.append(reward)
        
        # Compute prediction error
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta @ context
        except np.linalg.LinAlgError:
            pred = 0.0
            
        err = reward - pred
        self.err_hist[arm].append(err * err)
        
        # Adapt gamma
        if len(self.err_hist[arm]) > 5 and len(self.reward_hist) > 5:
            # Estimate scale
            # Use median absolute deviation or just mean abs
            # Median is more robust to outliers
            scale = np.median(np.abs(list(self.reward_hist))) + 1e-6
            mean_sq_err = np.mean(self.err_hist[arm])
            # Ratio of observed variance to expected scale
            # If model is good, err^2 ~ sigma^2.
            # We use scale^2 as a proxy for sigma^2 * (1 + something)
            # If ratio is high, model is bad.
            ratio = mean_sq_err / (scale * scale)
            
            if ratio > 2.0:
                # High error, suspect change, forget faster
                self.gammas[arm] = max(self.gamma_min, self.gammas[arm] - self.gamma_step)
            elif ratio < 0.5:
                # Low error, stable, remember longer
                self.gammas[arm] = min(self.gamma_max, self.gammas[arm] + self.gamma_step)
        
        # Update statistics
        gamma = self.gammas[arm]
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
