import numpy as np

class Policy:
    """Discounted Linear UCB for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Decay factor
        self.lam = 1.0     # Regularization
        self.beta = 1.0    # Confidence parameter
        
        # Initialize A and b for each arm
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, context):
        x = np.array(context)
        scores = []
        
        for a in range(self.n):
            try:
                # Compute mean: mu = A^-1 b
                mu = np.linalg.solve(self.A[a], self.b[a])
                
                # Compute confidence term: sqrt(beta * x^T A^-1 x)
                # We need to solve A v = x to get v = A^-1 x
                # Then x^T A^-1 x = x @ v
                v = np.linalg.solve(self.A[a], x)
                conf = self.beta * np.sqrt(x @ v)
                
                score = mu @ x + conf
                scores.append(score)
            except np.linalg.LinAlgError:
                # Fallback
                scores.append(np.random.rand())

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.array(context)
        # Update with decay
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + reward * x
