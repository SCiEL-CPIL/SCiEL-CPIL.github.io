"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with arm-specific bias terms and a forgetting factor.
    
    Instead of maintaining a separate full linear model per arm, we maintain:
      - A single global model (A_global, b_global) capturing shared context features
      - Per-arm bias terms (bias[arm]) capturing arm-specific offsets
    
    The predicted reward for arm a given context x is:
      theta_global @ x + bias[a]
    
    Exploration bonus is computed from the global model's uncertainty.
    This reduces parameter count and leverages cross-arm correlations.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global linear model
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        
        # Arm-specific biases (scalar per arm)
        self.bias = np.zeros(n_arms)
        
        # Counters for bias estimation
        self.count = np.zeros(n_arms)
        
        # Forgetting factor
        self.forget = 0.99
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Track recent rewards per arm for adaptive bias
        self.recent_rewards = [np.zeros(0) for _ in range(n_arms)]
        self.max_recent = 20

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute global theta
        try:
            inv_A = np.linalg.inv(self.A)
            theta = inv_A @ self.b
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
            inv_A = np.eye(self.d)
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Global linear prediction
            pred = theta @ context
            # Add arm-specific bias
            pred += self.bias[a]
            # Exploration bonus based on global uncertainty
            bonus = self.alpha * np.sqrt(max(0.0, context @ inv_A @ context))
            scores[a] = pred + bonus
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Update global model with forgetting
        self.A = self.forget * self.A + np.outer(context, context)
        self.b = self.forget * self.b + reward * context
        
        # Update arm-specific bias
        self.count[arm] += 1
        # Exponentially weighted moving average for bias
        if self.count[arm] == 1:
            self.bias[arm] = reward
        else:
            # Use a smaller forgetting factor for bias to adapt quickly
            beta = 0.95
            self.bias[arm] = beta * self.bias[arm] + (1 - beta) * reward
        
        # Keep recent rewards for potential adaptive tuning
        self.recent_rewards[arm] = np.concatenate([self.recent_rewards[arm], [reward]])
        if len(self.recent_rewards[arm]) > self.max_recent:
            self.recent_rewards[arm] = self.recent_rewards[arm][-self.max_recent:]
# EVOLVE-BLOCK-END
