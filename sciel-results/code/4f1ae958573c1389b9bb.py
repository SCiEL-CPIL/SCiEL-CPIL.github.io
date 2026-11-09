"""
Non-stationary linear contextual bandit policy using Discounted UCB.
This approach replaces the complex parameter-optimized Thompson Sampling
with a robust Upper Confidence Bound strategy using a forgetting factor.
This allows for faster per-step computation and more reliable adaptation
to non-stationarity without relying on synthetic tuning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.start_time = time.time()
        
        # Hyperparameters for Discounted UCB
        # gamma: Forgetting factor. 0.99 means ~50 steps half-life.
        # beta: Exploration bonus coefficient. Higher = more exploration.
        # In non-stationary settings, we need higher exploration to detect changes.
        self.gamma = 0.99
        self.beta = 2.5
        
        # Initialize per-arm models
        # A: Matrix for inverse covariance (regularized)
        # b: Vector for linear system
        eye = np.eye(dim)
        self.A = [eye.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Pre-compute inverse of initial A (which is identity) for speed if needed, 
        # but we will update A and solve each time.
        # For small dim (e.g., 5-10), np.linalg.solve is fast enough.
        
    def select(self, context):
        """
        Selects an arm using UCB1 rule with discounted covariance.
        """
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1)
            
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            A = self.A[a]
            b = self.b[a]
            
            # Solve for theta: theta = A^{-1} b
            # We use solve for numerical stability
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with regularization/initialization)
                theta = np.zeros(self.dim)
            
            # Predicted mean
            mean = float(np.dot(theta, context))
            
            # Confidence bound: beta * sqrt(x^T A^{-1} x)
            # We need A^{-1} x. Let v = A^{-1} x.
            # Then x^T A^{-1} x = x^T v = dot(x, v).
            try:
                v = np.linalg.solve(A, context)
            except np.linalg.LinAlgError:
                v = np.zeros(self.dim)
            
            variance_term = float(np.dot(context, v))
            if variance_term < 0:
                variance_term = 0.0 # Numerical safety
            
            std = np.sqrt(variance_term)
            score = mean + self.beta * std
            
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
        # A_new = g * A_old + x x^T
        # b_new = g * b_old + r x
        
        self.A[a] = g * self.A[a] + np.outer(context, context)
        self.b[a] = g * self.b[a] + r * context
        
# EVOLVE-BLOCK-END
