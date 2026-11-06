import numpy as np
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 100
        self.reg = 1.0
        
        # Per-arm history and stats
        self.history = [deque(maxlen=self.window_size) for _ in range(self.n)]
        self.S = [self.reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        
        # Exploration parameter
        self.alpha = 1.0

    def select(self, context):
        scores = np.zeros(self.n)
        for a in range(self.n):
            mu = context @ self.theta[a]
            
            # Compute uncertainty from current S
            # S = sum(x x^T) + reg * I
            # Covariance of theta is proportional to S^{-1}
            try:
                S_inv = np.linalg.inv(self.S[a])
                var = np.dot(context, S_inv @ context)
                uncertainty = np.sqrt(var)
            except np.linalg.LinAlgError:
                uncertainty = 1.0
            
            scores[a] = mu + self.alpha * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        h = self.history[arm]
        
        # If window is full, remove oldest
        if len(h) == self.window_size:
            old_x, old_r = h[0]
            self.S[arm] -= np.outer(old_x, old_x)
            self.b[arm] -= old_x * old_r
        
        # Add new
        h.append((context, reward))
        self.S[arm] += np.outer(context, context)
        self.b[arm] += context * reward
        
        # Solve for theta
        try:
            self.theta[arm] = np.linalg.solve(self.S[arm], self.b[arm])
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular (should be rare with reg)
            self.theta[arm] = np.zeros(self.d)
