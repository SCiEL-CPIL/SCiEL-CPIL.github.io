"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear bandit using Recursive Least Squares (RLS) with Exponential Forgetting.
    
    This approach maintains a per-arm estimate of the weight vector (theta) and its 
    inverse covariance matrix (P). The forgetting factor beta allows the model to 
    adapt to non-stationary environments by discounting older observations.
    Exploration is handled via a LinUCB-style bonus term.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.beta = 0.95  # Forgetting factor
        self.alpha = 1.0  # Exploration coefficient
        self.lam = 1.0    # Initial regularization
        
        # Per-arm state: theta (weights) and P (inverse covariance)
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        self.P = [self.lam * np.eye(self.d) for _ in range(self.n)]
        
        # Track number of updates per arm for potential alpha adjustment
        self.arm_counts = np.zeros(self.n)

    def select(self, context):
        """Select arm using optimistic linear regression with RLS."""
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Prediction
            mu = context @ self.theta[a]
            
            # Uncertainty
            # variance = context @ P @ context
            v = self.P[a] @ context
            uncertainty = np.sqrt(np.dot(context, v))
            
            scores[a] = mu + self.alpha * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the RLS model for the selected arm."""
        self.arm_counts[arm] += 1
        
        # Sherman-Morrison update with forgetting factor
        # K = P x / (beta + x^T P x)
        # theta_new = theta_old + K (r - x^T theta_old)
        # P_new = (1/beta) (P_old - K x^T P_old)
        
        P = self.P[arm]
        theta = self.theta[arm]
        
        Px = P @ context
        denom = self.beta + np.dot(context, Px)
        
        if denom < 1e-6:
            # Avoid division by zero
            denom = 1e-6
            
        K = Px / denom
        
        # Update theta
        error = reward - np.dot(context, theta)
        self.theta[arm] = theta + K * error
        
        # Update P
        self.P[arm] = (P - np.outer(K, context) @ P) / self.beta
        
# EVOLVE-BLOCK-END
