"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

class Policy:
    """Sliding Window Linear Thompson Sampling for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Window size for sliding window
        self.window_size = 50
        # Store history for each arm
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        # Regularization parameter (prior precision)
        self.reg = 1.0
        # Noise variance assumption
        self.noise_var = 1.0

    def select(self, context):
        scores = []
        for a in range(self.n):
            hist = self.history[a]
            
            if len(hist) == 0:
                # No data, sample from prior N(0, I)
                theta = np.random.normal(0, 1, self.d)
            else:
                # Extract data
                contexts = np.array([h[0] for h in hist])
                rewards = np.array([h[1] for h in hist])
                
                # Compute X^T X and X^T y
                A = contexts.T @ contexts + self.reg * np.eye(self.d)
                b = contexts.T @ rewards
                
                # Compute mean and covariance
                # Mean = A^-1 b
                # Cov = A^-1 (assuming noise_var=1)
                try:
                    A_inv = np.linalg.inv(A)
                    mu = A_inv @ b
                    cov = A_inv
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    A_inv = np.linalg.pinv(A)
                    mu = A_inv @ b
                    cov = A_inv
                
                # Sample from N(mu, cov)
                try:
                    L = np.linalg.cholesky(cov)
                    z = np.random.normal(0, 1, self.d)
                    theta = mu + L @ z
                except np.linalg.LinAlgError:
                    # Fallback if cov is not PD
                    cov_jittered = cov + 1e-6 * np.eye(self.d)
                    L = np.linalg.cholesky(cov_jittered)
                    z = np.random.normal(0, 1, self.d)
                    theta = mu + L @ z
            
            scores.append(np.dot(theta, context))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.history[arm].append((context, reward))
# EVOLVE-BLOCK-END
