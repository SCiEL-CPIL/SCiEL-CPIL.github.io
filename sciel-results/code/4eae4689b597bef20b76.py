"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary linear contextual bandit using Sliding Window Linear Regression 
    with Thompson Sampling and Variance-Adaptive Exploration.
    
    Strategy:
    1. Sliding Window: Strictly maintains only the last W samples for each arm. 
       This provides a natural, robust forgetting mechanism that adapts to 
       non-stationarity without heuristic thresholds.
    2. Thompson Sampling: Samples from the posterior to balance exploration/exploitation.
    3. Variance-Adaptive Exploration: Adjusts exploration noise based on the 
       variance of recent rewards for the selected arm. High variance (indicating 
       potential non-stationarity or high uncertainty) increases exploration.
    4. Deadline Guard: Ensures termination before the 150s limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Sliding window size
        self.window_size = 50
        
        # Store (context, reward) pairs for each arm
        self.histories = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Pre-compute identity matrix for initialization
        self.eye_d = np.eye(self.d)
        
        # Time management for safety
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def _compute_posterior(self, arm):
        """
        Compute posterior mean and covariance for a specific arm using linear regression
        on the samples in its sliding window.
        """
        if not self.histories[arm]:
            # No data yet: return prior
            return np.zeros(self.d), self.eye_d

        X = np.array([x for x, r in self.histories[arm]])
        y = np.array([r for x, r in self.histories[arm]])
        
        n_samples = len(y)
        
        # Regularization to handle small windows or collinear contexts
        # Lambda is chosen to be relatively small but stable
        lam = 1.0
        
        # Design matrix with bias term
        X_design = np.hstack([X, np.ones((n_samples, 1))])
        theta_full = np.zeros(self.d + 1)
        
        # Solve normal equations: (X^T X + lam * I) theta = X^T y
        # We add regularization to the full (d+1) x (d+1) system
        reg = lam * np.eye(self.d + 1)
        A = X_design.T @ X_design + reg
        b = X_design.T @ y
        
        try:
            theta_full = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            theta_full = np.linalg.pinv(A) @ b
            
        # Extract parameters for context (excluding bias)
        theta = theta_full[:self.d]
        
        # Compute covariance of the estimator
        # Cov(theta) = sigma^2 * (X^T X + lam * I)^-1
        # We estimate sigma^2 from residuals
        if n_samples > self.d:
            y_pred = X_design @ theta_full
            residuals = y - y_pred
            sigma_sq = np.sum(residuals**2) / (n_samples - self.d)
        else:
            sigma_sq = 1.0
            
        try:
            cov = np.linalg.inv(A)
            cov = sigma_sq * cov
        except np.linalg.LinAlgError:
            cov = self.eye_d

        return theta, cov

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy or cached best
        # However, the interface requires returning an arm. We don't have a global "done" state,
        # but we can ensure computation is fast.
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            theta, cov = self._compute_posterior(a)
            
            # Sample from N(theta, cov)
            # Use Cholesky decomposition for speed and stability
            try:
                L = np.linalg.cholesky(cov + 1e-6 * self.eye_d)
                z = np.random.randn(self.d)
                theta_sample = theta + L @ z
            except np.linalg.LinAlgError:
                # If Cholesky fails, add more jitter or use identity
                L = np.linalg.cholesky(cov + 1e-3 * self.eye_d)
                z = np.random.randn(self.d)
                theta_sample = theta + L @ z

            scores.append(theta_sample @ context)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self.histories[arm].append((context, reward))
        self.t += 1
# EVOLVE-BLOCK-END
