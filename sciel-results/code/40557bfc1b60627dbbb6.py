import numpy as np

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lambda_forget = 0.95
        self.sigma2 = 1.0
        
        # Initialize Precision (V) and Mean (mu) for each arm
        # Start with very low precision (high uncertainty)
        self.V = [1e-6 * np.eye(self.d) for _ in range(self.n)]
        self.mu = [np.zeros(self.d) for _ in range(self.n)]
        
        # Precompute inverses? No, V changes every update.
        # We will compute P = inv(V) for sampling.

    def select(self, context):
        samples = np.zeros(self.n)
        for a in range(self.n):
            # Compute covariance P from precision V
            try:
                P = np.linalg.inv(self.V[a])
            except np.linalg.LinAlgError:
                P = np.eye(self.d)
            
            # Sample theta from N(mu, P)
            # To ensure positive definiteness for sampling, add small jitter if needed
            theta = np.random.multivariate_normal(self.mu[a], P)
            samples[a] = context @ theta
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        V = self.V[arm]
        mu = self.mu[arm]
        
        # Update Precision: V_new = lambda * V_old + (1/sigma2) * x x^T
        V_new = self.lambda_forget * V + (1.0 / self.sigma2) * np.outer(context, context)
        
        # Update Mean: mu_new = P_new * (lambda * V_old * mu_old + (1/sigma2) * x * r)
        # Note: The term (lambda * V_old * mu_old) is the "prior information" vector.
        # Let's compute the numerator vector first.
        prior_info = self.lambda_forget * (V @ mu) + (1.0 / self.sigma2) * context * reward
        
        # Compute P_new
        try:
            P_new = np.linalg.inv(V_new)
        except np.linalg.LinAlgError:
            P_new = np.eye(self.d)
            
        mu_new = P_new @ prior_info
        
        self.V[arm] = V_new
        self.mu[arm] = mu_new
