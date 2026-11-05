import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.gamma = 0.90
        self.alpha = 1.0  # Exploration coefficient
        self.means = np.zeros(n_arms)
        # Initialize effective counts to 1.0 to ensure initial exploration and avoid div by 0
        self.effective_counts = np.ones(n_arms)

    def select(self, context):
        # Ensure counts are not zero to avoid division by zero
        counts = np.maximum(self.effective_counts, 1e-6)
        
        # Exploration bonus: inversely proportional to sqrt of effective count
        # This is a form of UCB that adapts to non-stationarity via the decaying counts
        bonus = self.alpha / np.sqrt(counts)
        scores = self.means + bonus
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Decay all effective counts
        self.effective_counts *= self.gamma
        
        # Increment the effective count for the selected arm
        self.effective_counts[arm] += 1.0
        
        # Update the mean reward estimate for the selected arm
        self.means[arm] = self.gamma * self.means[arm] + (1 - self.gamma) * reward
