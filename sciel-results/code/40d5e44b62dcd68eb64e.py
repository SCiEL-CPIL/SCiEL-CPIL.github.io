import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.gamma = 0.90
        self.alpha = 1.0
        self.ema_mean = np.zeros(n_arms)
        # Initialize counts to 1.0 to avoid division by zero and provide initial exploration bonus
        self.counts = np.ones(n_arms)
        
    def select(self, context):
        # Calculate total discounted count for the log term
        total_count = np.sum(self.counts)
        if total_count <= 0:
            total_count = 1.0
            
        # Exploration bonus based on discounted counts
        # This encourages playing arms that have been "forgotten" (low count)
        bonus = self.alpha * np.sqrt(np.log(total_count) / self.counts)
        
        scores = self.ema_mean + bonus
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        r = reward
        # Update EMA of the mean reward
        self.ema_mean[arm] = self.gamma * self.ema_mean[arm] + (1 - self.gamma) * r
        
        # Update discounted count for the arm
        self.counts[arm] = self.gamma * self.counts[arm] + 1.0
