import numpy as np

class Policy:
    """Discounted Linear Thompson Sampling for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Decay factor
        self.lam = 1.0     # Regularization
        
        # Initialize A and b for each arm
        # A represents the precision matrix (inverse covariance)
        # b represents A * mean
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, context):
        x = np.array(context)
        scores = []
        
        for a in range(self.n):
            try:
                # Compute mean: mu = A^-1 b
                mu = np.linalg.solve(self.A[a], self.b[a])
                
                # Compute sample theta from N(mu, A^-1)
                # We need to sample from multivariate normal with cov A^-1
                # Let A = L L^T
                # Then A^-1 = L^-T L^-1
                # Sample z ~ N(0, I)
                # theta = mu + L^-T z
                # Solve L^T w = z for w
                L = np.linalg.cholesky(self.A[a])
                z = np.random.randn(self.d)
                w = np.linalg.solve(L.T, z)
                theta_sample = mu + w
                
                score = theta_sample @ x
                scores.append(score)
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                try:
                    mu = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(mu @ x)
                except:
                    scores.append(np.random.rand())

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.array(context)
        # Update with decay
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + reward * x
