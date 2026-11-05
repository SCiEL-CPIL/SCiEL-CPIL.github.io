"""Rewritten program for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Uses a sliding window approach with exponential forgetting, but with
    adaptive beta based on reward change detection. Also uses a more
    robust exploration strategy.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive parameters
        self.beta = 0.95
        self.alpha = 1.2
        
        # Tracking
        self.t = 0
        self.reward_history = []
        self.reward_window = 20
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Per-arm statistics for change detection
        self.arm_rewards = {a: [] for a in range(n_arms)}
        self.arm_reward_mean = {a: 0.0 for a in range(n_arms)}
        self.arm_reward_std = {a: 1.0 for a in range(n_arms)}
        
        # Global reward tracking
        self.global_reward_mean = 0.0
        self.global_reward_std = 1.0

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fast fallback: pick arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                mu = theta @ context
                
                inv_A_ctx = np.linalg.solve(self.A[a], context)
                var = context @ inv_A_ctx
                if var < 0:
                    var = 0.0
                uc = self.alpha * np.sqrt(var)
                
                scores.append(mu + uc)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for minimal update
        if time.time() > self.deadline:
            self.A[arm] = self.beta * self.A[arm]
            self.b[arm] = self.beta * self.b[arm]
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context
            return

        # Track per-arm rewards
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > self.reward_window:
            self.arm_rewards[arm].pop(0)
        
        # Update global reward statistics
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)
        
        if len(self.reward_history) >= 5:
            self.global_reward_mean = np.mean(self.reward_history)
            self.global_reward_std = np.std(self.reward_history)
        
        # Per-arm statistics
        if len(self.arm_rewards[arm]) >= 3:
            self.arm_reward_mean[arm] = np.mean(self.arm_rewards[arm])
            self.arm_reward_std[arm] = np.std(self.arm_rewards[arm]) if len(self.arm_rewards[arm]) > 1 else 1.0
        
        # Adaptive beta: detect if rewards are changing
        # If reward differs significantly from recent mean, increase forgetting
        change_signal = 0.0
        if len(self.reward_history) >= 10:
            recent_mean = np.mean(self.reward_history[-10:])
            change_signal = abs(reward - recent_mean) / (self.global_reward_std + 1e-6)
        
        # Also check per-arm change
        if len(self.arm_rewards[arm]) >= 5:
            arm_mean = self.arm_reward_mean[arm]
            arm_change = abs(reward - arm_mean) / (self.arm_reward_std[arm] + 1e-6)
            change_signal = max(change_signal, arm_change)
        
        # Map change signal to beta
        # High change -> lower beta (faster forgetting)
        # Low change -> higher beta (slower forgetting)
        # Beta range: [0.85, 0.99]
        if change_signal > 3.0:
            target_beta = 0.85
        elif change_signal > 1.5:
            target_beta = 0.90
        elif change_signal > 0.5:
            target_beta = 0.95
        else:
            target_beta = 0.99
        
        # Smoothly adjust beta
        self.beta = 0.8 * self.beta + 0.2 * target_beta
        
        # Adaptive alpha: more exploration when uncertainty is high or changes detected
        # Base decay
        self.alpha = max(0.8, self.alpha * 0.998)
        
        # Boost exploration if change detected
        if change_signal > 2.0:
            self.alpha = min(2.0, self.alpha * 1.1)
        
        # Update model with forgetting
        self.A[arm] = self.beta * self.A[arm]
        self.b[arm] = self.beta * self.b[arm]
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
# EVOLVE-BLOCK-END
