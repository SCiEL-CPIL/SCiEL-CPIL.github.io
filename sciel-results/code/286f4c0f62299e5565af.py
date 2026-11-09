import numpy as np

class Policy:
    """
    Discounted Linear Thompson Sampling for non-stationary linear bandits.
    
    This policy maintains a Bayesian posterior over the weight vector for each arm.
    The posterior is updated with a discount factor to handle non-stationarity.
    At each step, it samples from the posterior to make an exploratory/exploitative decision.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.98  # Discount factor for forgetting
        self.lam = 1.0     # Regularization parameter
        
        # Initialize posterior parameters for each arm
        # A is the precision matrix (inverse covariance)
        # b is the momentum vector
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Pre-allocate space for Cholesky factors to avoid repeated allocation overhead
        # Note: We recompute Cholesky every step because A changes.
        
    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        best_arm = 0
        best_reward = -np.inf
        
        for a in range(self.n):
            # Compute mean theta = A^-1 b
            # Use solve for numerical stability: A theta = b
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with regularization)
                theta = np.zeros(self.d)
            
            # Compute covariance Sigma = A^-1
            # We need to sample from N(theta, Sigma).
            # Efficient way: Cholesky decomposition of Sigma.
            # Sigma = A^-1. Let L_A be Cholesky of A (A = L_A L_A^T).
            # Then Sigma = L_A^-T L_A^-.
            # Sampling z ~ N(0, I), then sample = theta + L_A^-T z.
            
            try:
                L_A = np.linalg.cholesky(self.A[a])
                # Solve L_A w = z for w, then sample = theta + w
                # Wait, if A = L L^T, then A^-1 = L^-T L^-.
                # Let z ~ N(0, I).
                # We want v ~ N(0, A^-1).
                # v = L^-T z.
                # Solve L^T v = z => v = L^-T z.
                # np.linalg.solve(L_A.T, z) gives v.
                
                z = np.random.normal(size=self.d)
                v = np.linalg.solve(L_A.T, z)
                theta_sample = theta + v
                
            except np.linalg.LinAlgError:
                # Fallback to inverse if Cholesky fails
                try:
                    Sigma = np.linalg.inv(self.A[a])
                    theta_sample = np.random.multivariate_normal(theta, Sigma)
                except np.linalg.LinAlgError:
                    theta_sample = theta

            reward_est = np.dot(theta_sample, context)
            
            if reward_est > best_reward:
                best_reward = reward_est
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Apply discount to previous statistics
        self.A[arm] = self.gamma * self.A[arm]
        self.b[arm] = self.gamma * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
