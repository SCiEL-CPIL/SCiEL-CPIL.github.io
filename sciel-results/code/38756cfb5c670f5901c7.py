import numpy as np

class Policy:
    """Discounted Linear Thompson Sampling for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.98  # Decay factor, faster forgetting than seed
        self.lam = 1.0     # Regularization
        
        # Initialize A and b for each arm
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Pre-compute inverses? No, A changes every step.
        # We will compute inv(A) on the fly.

    def select(self, context):
        scores = []
        # Ensure context is a numpy array for dot products
        x = np.array(context)
        
        for a in range(self.n):
            # Compute mean and covariance of the posterior
            # theta ~ N(mu, Sigma)
            # mu = A^-1 b
            # Sigma = A^-1
            
            # Solve A theta = b for mu, and A Sigma = I for Sigma
            # Using np.linalg.solve is more stable than inv
            try:
                mu = np.linalg.solve(self.A[a], self.b[a])
                # For sampling from multivariate normal, we need the covariance matrix
                # Sigma = A^-1
                # We can use np.linalg.inv or solve A X = I
                Sigma = np.linalg.inv(self.A[a])
                
                # Sample theta
                # np.random.multivariate_normal(mean, cov, size=1)
                # Note: In non-stationary settings, we might want to add some noise
                # to the mean to encourage exploration if the model is stale?
                # Standard TS handles this via variance.
                
                theta = np.random.multivariate_normal(mu, Sigma, size=1)[0]
                score = theta @ x
                scores.append(score)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular (unlikely with regularization)
                scores.append(np.random.rand())

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.array(context)
        # Update with decay
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + reward * x
