"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Discounted Linear Bandit with UCB."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Forgetting factor
        self.lambda_reg = 1.0 # Initial regularization
        self.beta = 1.5   # Exploration parameter
        
        # Initialize sufficient statistics for each arm
        # A[a] is the covariance matrix, b[a] is the reward vector
        self.A = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(self.n)]
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            pred = self.theta[a] @ context
            # Variance term: context^T A_inv context
            var = context @ self.A_inv[a] @ context
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Decay previous statistics
        self.A[arm] *= self.gamma
        self.b[arm] *= self.gamma
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update inverse and theta
        # Use Cholesky or just inv for simplicity, d is likely small
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.lambda_reg
            
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        
        self.t += 1
# EVOLVE-BLOCK-END
