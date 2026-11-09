"""Non-stationary linear contextual bandit policy using a global model with arm-encoded features."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with arm-encoded features and forgetting factor.
    
    Features: [context, one-hot(arm)] of dimension dim + n_arms.
    A single global A, b pair models reward = theta^T @ [context, one-hot(arm)].
    Uses LinUCB-style exploration with a forgetting factor for non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_total = dim + n_arms  # augmented feature dimension
        
        # Global model parameters
        self.A = np.eye(self.d_total)
        self.b = np.zeros(self.d_total)
        
        # Forgetting factor for non-stationarity
        self.gamma = 0.99
        
        # Exploration parameter
        self.alpha = 1.0

    def _build_feature(self, context, arm):
        """Construct augmented feature vector: [context, one-hot(arm)]."""
        feature = np.zeros(self.d_total)
        feature[:self.d] = context
        feature[self.d + arm] = 1.0
        return feature

    def select(self, context):
        """Select arm using UCB scores from the global model."""
        # Compute inverse once for efficiency
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d_total)
        
        scores = []
        for a in range(self.n):
            feature = self._build_feature(context, a)
            # Predicted reward
            pred = feature @ (A_inv @ self.b)
            # Uncertainty bonus
            uncertainty = self.alpha * np.sqrt(feature @ A_inv @ feature)
            scores.append(pred + uncertainty)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global model with observed reward."""
        feature = self._build_feature(context, arm)
        
        # Apply forgetting factor
        self.A = self.gamma * self.A + np.outer(feature, feature)
        self.b = self.gamma * self.b + reward * feature
# EVOLVE-BLOCK-END
