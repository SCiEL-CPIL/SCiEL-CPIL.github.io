"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices A and mean vectors b
        # A is the precision matrix (inverse covariance)
        # b is the sufficient statistic for the mean
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor: 0.99 provides a balance between memory and adaptability
        # A smaller lambda (e.g., 0.9) adapts faster but has higher variance
        # A larger lambda (e.g., 0.995) adapts slower but is more stable
        self.alpha = 0.99
        
        # Assumed noise variance for Thompson Sampling
        # This scales the uncertainty. If rewards are standardized, 1.0 is reasonable.
        self.sigma_sq = 1.0
        
        # Start time for deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback to a simple heuristic if we are about to time out
            # This should rarely happen, but ensures we return something valid
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Compute the posterior mean and covariance
            # theta ~ N(A^-1 b, A^-1 sigma^2)
            
            # Solve A theta = b for theta
            # Using np.linalg.solve is more stable than inverse
            try:
                theta_mean = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular (should not happen with eye init + outer product)
                theta_mean = np.zeros(self.d)

            # Sample theta from the posterior
            # We need to sample from N(theta_mean, sigma^2 * A^-1)
            # Let L be such that L L^T = A^-1. Then theta = theta_mean + sigma * L z
            # Instead of inverting A, we can use Cholesky decomposition of A.
            # If A = L L^T, then A^-1 = (L^-1)^T L^-1
            # So sqrt(A^-1) = (L^-1)^T
            # We can sample z ~ N(0, I), then theta = theta_mean + sigma * (L^-1)^T z
            
            try:
                # Cholesky decomposition of A
                L = np.linalg.cholesky(self.A[a])
                # Solve L y = sigma * z for y, then theta = theta_mean + L^-T y?
                # Actually, if A = L L^T, then A^-1 = (L^T)^-1 L^-1
                # Covariance is sigma^2 A^-1.
                # Let v ~ N(0, I). Then theta = theta_mean + sigma * A^-1/2 v.
                # A^-1/2 = (L^T)^-1.
                # So we solve L^T w = sigma * v, then theta = theta_mean + w.
                # Or simpler: solve L u = sigma * v, then w = L^-T u? No.
                # Let's use the standard method:
                # 1. Cholesky of A: A = L L^T
                # 2. Solve L y = sigma * z (where z ~ N(0, I))
                # 3. Then theta = theta_mean + L^-T y? 
                # Actually, if we want to sample from N(mu, Sigma) where Sigma = sigma^2 A^-1.
                # Let Sigma = (sigma L^-T) (sigma L^-1) ? No.
                # Sigma = sigma^2 A^-1.
                # Let L be Cholesky of A. A = L L^T.
                # A^-1 = (L^T)^-1 L^-1.
                # So Sigma = sigma^2 (L^T)^-1 L^-1.
                # This is of the form M M^T where M = sigma (L^T)^-1.
                # So we need to sample M z.
                # M z = sigma (L^T)^-1 z.
                # Let w = (L^T)^-1 z. This is equivalent to solving L^T w = z.
                # Then sample = sigma * w.
                
                z = np.random.randn(self.d)
                # Solve L^T w = z
                w = np.linalg.solve(L.T, z)
                theta_sample = theta_mean + self.sigma_sq * w
            except np.linalg.LinAlgError:
                # Fallback to mean if Cholesky fails
                theta_sample = theta_mean

            score = theta_sample @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Apply forgetting factor
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
# EVOLVE-BLOCK-END
