import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.dim = dim
        self.gamma = 0.95
        self.alpha = 1.0
        # Per-arm linear models: weight vectors and inverse covariance matrices
        self.theta = np.zeros((n_arms, dim))
        self.A_inv = np.eye(dim) * 0.1  # Regularized inverse covariance
        self.A = np.eye(dim) * 10.0     # Inverse of A_inv
        self.b = np.zeros((n_arms, dim))  # Weighted sum of context*reward
        
    def select(self, context):
        ctx = np.array(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for arm in range(self.n):
            # Predicted reward
            pred = self.theta[arm] @ ctx
            # Uncertainty: sqrt(ctx^T * A_inv * ctx)
            uncertainty = np.sqrt(max(0.0, ctx @ self.A_inv @ ctx))
            scores[arm] = pred + self.alpha * uncertainty
        return int(np.argmax(scores))
    
    def update(self, context, arm, reward):
        ctx = np.array(context, dtype=np.float64)
        # Update A and b with exponential forgetting
        self.A = self.gamma * self.A + np.outer(ctx, ctx)
        self.b[arm] = self.gamma * self.b[arm] + ctx * reward
        # Update inverse covariance using Sherman-Morrison or direct inversion
        # For small dim, direct inversion is fine
        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.eye(self.dim) * 0.1
        # Update theta
        self.theta[arm] = self.A_inv @ self.b[arm]
