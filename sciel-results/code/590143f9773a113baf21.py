"""
Adaptive Sliding Window LinUCB for Non-Stationary Linear Bandits.

Uses a sliding window of recent observations for each arm to maintain a regularized
linear regression model. This approach allows the policy to completely forget
old data after a fixed window, providing faster adaptation to non-stationarity
compared to exponential forgetting. Incremental updates to the design matrix and
response vector ensure O(d^2) cost for updates, with O(d^3) cost for the matrix
inverse which is only recomputed for the updated arm.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Sliding window parameters
        self.window_size = 30  # Number of recent samples to keep per arm
        self.reg = 1.0         # Regularization parameter (ridge)
        self.alpha = 1.5       # Exploration parameter for UCB
        
        # Data storage for sliding window
        self.contexts = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.rewards = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Linear regression state: A = reg*I + sum(x x^T), b = sum(r x)
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached inverse and weights for fast select
        self.A_inv = [np.linalg.inv(self.A[i]) for i in range(n_arms)]
        self.theta = [self.A_inv[i] @ self.b[i] for i in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Predicted mean
            pred = context @ self.theta[a]
            
            # Confidence term: sqrt(x^T A_inv x)
            # Compute efficiently
            Ax = self.A_inv[a] @ context
            conf = np.sqrt(np.dot(context, Ax))
            
            # UCB score
            scores[a] = pred + self.alpha * conf
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # If window is full, remove the oldest sample
        if len(self.contexts[arm]) == self.window_size:
            old_ctx = self.contexts[arm][0]
            old_rew = self.rewards[arm][0]
            
            # Incrementally subtract from A and b
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_rew * old_ctx
            
        # Add new sample
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
        
        # Incrementally add to A and b
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Recompute inverse and weights for the updated arm only
        # Ensure numerical stability
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            # If singular, add small regularization
            self.A[arm] += 1e-6 * np.eye(self.dim)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
# EVOLVE-BLOCK-END
