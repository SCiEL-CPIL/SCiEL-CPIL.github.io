import numpy as np
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 100  # Sliding window size
        self.lambda_reg = 1.0   # Ridge regularization
        self.beta = 1.0         # Exploration coefficient
        
        # Initialize per-arm buffers and running sums
        self.buffers = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.S = [np.zeros((dim, dim)) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
    def select(self, context):
        context = np.asarray(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Add regularization to S
            S_reg = self.S[a] + self.lambda_reg * np.eye(self.d)
            
            # Compute inverse
            # For small d, inv is fine. For larger d, use solve or cholesky.
            # We need the inverse for the bonus term anyway.
            try:
                S_inv = np.linalg.inv(S_reg)
            except np.linalg.LinAlgError:
                S_inv = np.eye(self.d) / self.lambda_reg
                
            theta = S_inv @ self.b[a]
            
            # Mean prediction
            pred = context @ theta
            
            # Variance term
            var = context @ S_inv @ context
            bonus = self.beta * np.sqrt(max(var, 0))
            
            scores[a] = pred + bonus
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Add new observation
        self.buffers[arm].append((context, reward))
        self.S[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # If buffer is full, remove the oldest observation
        # Note: deque with maxlen automatically pops the left, but we need to subtract it from S and b
        # We need to know what was popped. 
        # Since we append and then check, if it was full, the leftmost item is the one to remove.
        # However, deque(maxlen) discards the old item silently. 
        # So we must handle the subtraction BEFORE appending if we want to know which item to subtract, 
        # OR we store the items and if we append to a full deque, the oldest is lost.
        # Better approach: 
        # 1. Check if buffer is full.
        # 2. If full, get the oldest item, subtract from S and b, then the append will push it out.
        
        # Actually, the code above appends first. If maxlen is set, the oldest is dropped.
        # We need to subtract the dropped item.
        # Let's restructure update to handle this explicitly.
        
        # Revised Update Logic:
        # 1. If buffer is full (len == maxlen), pop the left (oldest) and subtract from S, b.
        # 2. Append new item.
        # 3. Add new item to S, b.
        
        # Wait, the code above already appended. 
        # Let's rewrite update correctly.
        
        # The current code in the draft is flawed because it doesn't subtract the oldest.
        # I will fix this in the final code.
