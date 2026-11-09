"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """Thompson Sampling with sliding window for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.window_size = 100  # Fixed window for non-stationarity
        
        # Storage for recent (context, reward) pairs per arm
        self.histories = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]
        
        # Initial parameters (prior)
        # We use a ridge regression approach on the window
        # A_inv = (X^T X + lambda * I)^-1
        # b = X^T y
        # theta = A_inv * b
        # Cov = A_inv * sigma^2
        
        self.lambda_reg = 1.0
        self.sigma2 = 1.0
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit
        self.elapsed = 0.0

    def _compute_posterior(self, arm):
        """Compute mean and covariance for the linear model of a given arm."""
        hist = self.histories[arm]
        if len(hist) == 0:
            # Prior: mean 0, large covariance
            mean = np.zeros(self.dim)
            cov = np.eye(self.dim) * 10.0
            return mean, cov

        X = np.array([h[0] for h in hist])
        y = np.array([h[1] for h in hist])
        
        # Ridge regression
        A = X.T @ X + self.lambda_reg * np.eye(self.dim)
        b = X.T @ y
        
        # Solve for theta and covariance
        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            cov = A_inv * self.sigma2
        except np.linalg.LinAlgError:
            # Fallback if singular
            theta = np.zeros(self.dim)
            cov = np.eye(self.dim) * 10.0
            
        return theta, cov

    def select(self, context):
        # Deadline guard: if we are close to the limit, use a simple heuristic
        # to avoid expensive computations that might timeout
        if time.time() > self.deadline:
            # Return the arm with the most recent highest reward or just 0
            # A simple greedy based on last reward or random
            return int(np.random.randint(self.n_arms))

        context = np.array(context)
        scores = []
        
        for a in range(self.n_arms):
            theta, cov = self._compute_posterior(a)
            
            # Sample theta_sample from N(theta, cov)
            # For efficiency, we can just sample the score directly if dim is small
            # score = theta_sample @ context
            # theta_sample = theta + z, where z ~ N(0, cov)
            # score = theta @ context + z @ context
            # z @ context ~ N(0, context @ cov @ context)
            
            mean_score = theta @ context
            var_score = context @ cov @ context
            std_score = np.sqrt(max(var_score, 1e-8))
            
            # Sample from normal distribution
            sampled_score = np.random.normal(mean_score, std_score)
            scores.append(sampled_score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.array(context)
        self.histories[arm].append((context, reward))
        
        # Update elapsed time for deadline guard
        self.elapsed = time.time() - self.start_time
# EVOLVE-BLOCK-END
