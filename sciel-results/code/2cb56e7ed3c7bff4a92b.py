"""Non-stationary linear contextual bandit using global model + arm indicators + Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Uses a single global linear model where the feature vector is the context
    concatenated with a one-hot arm indicator. This allows sharing statistical
    strength across arms while still allowing different intercepts/behaviors.
    
    Exploration via Thompson Sampling: sample theta from posterior Gaussian,
    then pick the arm with highest predicted reward under the sampled theta.
    
    Non-stationarity handled via fixed exponential forgetting factor.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.total_dim = dim + n_arms  # context + one-hot arm indicator
        
        # Global model parameters
        self.lam = 1.0
        self.A = np.eye(self.total_dim) * self.lam
        self.b = np.zeros(self.total_dim)
        
        # Fixed forgetting factor
        self.beta = 0.98
        
        # Tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Cached Cholesky factor for speed
        self.L = np.linalg.cholesky(self.A)
        self.L_valid = True

    def _build_feature(self, context, arm):
        """Build augmented feature vector: [context, one_hot(arm)]"""
        feat = np.zeros(self.total_dim)
        feat[:self.d] = context
        feat[self.d + arm] = 1.0
        return feat

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fast fallback: use current mean theta
            try:
                theta = np.linalg.solve(self.A, self.b)
            except:
                theta = np.zeros(self.total_dim)
            
            scores = []
            for a in range(self.n):
                feat = self._build_feature(context, a)
                scores.append(theta @ feat)
            return int(np.argmax(scores))

        # Thompson Sampling: sample theta from posterior
        try:
            # Sample from N(theta_hat, (A^{-1}))
            # theta_hat = A^{-1} b
            # Sample: theta_hat + A^{-1/2} * z, where z ~ N(0, I)
            
            if not self.L_valid:
                self.L = np.linalg.cholesky(self.A)
                self.L_valid = True
            
            # Solve A x = b for theta_hat
            # Use Cholesky: A = L L^T, so L L^T x = b
            # L^T x = L^{-1} b, then x = (L^T)^{-1} (L^{-1} b)
            L_inv_b = np.linalg.solve(self.L, self.b)
            theta_hat = np.linalg.solve(self.L.T, L_inv_b)
            
            # Sample z ~ N(0, I)
            z = np.random.randn(self.total_dim)
            
            # Sample theta = theta_hat + L^{-1} z
            # Since A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
            # Covariance is A^{-1}, so sample is theta_hat + L^{-T} L^{-1} z
            # But easier: sample v ~ N(0, A^{-1}) by solving L^T w = z, then v = L^{-1} w
            # Actually: if A = L L^T, then A^{-1} = L^{-T} L^{-1}
            # So sample = L^{-T} L^{-1} z
            # Let's just do: solve L^T u = z, then sample_offset = L^{-1} u? No.
            # Simpler: solve L v = z, then offset = L^{-T} v? 
            # Let me think again. We want w ~ N(0, A^{-1}).
            # A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
            # So w = L^{-T} L^{-1} z where z ~ N(0,I)
            # Step 1: solve L y = z  => y = L^{-1} z
            # Step 2: solve L^T w = y => w = L^{-T} y = L^{-T} L^{-1} z
            y = np.linalg.solve(self.L, z)
            w = np.linalg.solve(self.L.T, y)
            
            theta_sampled = theta_hat + w
            
        except np.linalg.LinAlgError:
            # Fallback to mean
            try:
                theta_sampled = np.linalg.solve(self.A, self.b)
            except:
                theta_sampled = np.zeros(self.total_dim)
        
        # Score each arm using the sampled theta
        scores = np.empty(self.n)
        for a in range(self.n):
            feat = self._build_feature(context, a)
            scores[a] = theta_sampled @ feat
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for minimal update
        if time.time() > self.deadline:
            feat = self._build_feature(context, arm)
            self.A = self.beta * self.A
            self.b = self.beta * self.b
            self.A += np.outer(feat, feat)
            self.b += reward * feat
            self.L_valid = False
            return

        feat = self._build_feature(context, arm)
        
        # Update with exponential forgetting
        self.A = self.beta * self.A
        self.b = self.beta * self.b
        self.A += np.outer(feat, feat)
        self.b += reward * feat
        
        # Invalidate Cholesky cache
        self.L_valid = False
# EVOLVE-BLOCK-END
