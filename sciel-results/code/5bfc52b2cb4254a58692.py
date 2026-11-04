"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor and maintained inverse."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Maintain A^{-1} directly for O(d^2) updates instead of O(d^3) inversions
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor - start higher for better initial learning
        self.gamma = 0.99
        self.gamma_min = 0.92
        self.gamma_max = 0.999
        self.error_ema = 0.0
        self.error_alpha = 0.20
        
        # Adaptive exploration parameter
        self.C = 1.0
        self.C_min = 0.3
        self.C_max = 2.0
        self.reward_ema = 0.0
        self.reward_alpha = 0.10
        
        # Track recent rewards for better non-stationarity detection
        self.recent_rewards = []
        self.recent_reward_window = 50
        
        self.round_count = 0
        
        # Deadline guard: track start time
        self._start_time = time.time()
        self._deadline = 150.0  # seconds
        self._last_update_time = time.time()

    def select(self, context):
        # Deadline guard: if we're getting close to the limit, be conservative
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline - 5.0:
            # Return the arm with highest mean reward estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm
        
        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            # Add exploration bonus
            scores.append(theta @ context + self.C * np.sqrt(context @ inv @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        
        # Check deadline before doing expensive work
        if time.time() - self._start_time > self._deadline - 2.0:
            return
        
        # Track recent rewards for non-stationarity detection
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_reward_window:
            self.recent_rewards.pop(0)
        
        # Adaptive gamma: detect non-stationarity via reward variance
        if self.round_count > 20 and len(self.recent_rewards) >= 10:
            # Compute recent reward variance as non-stationarity indicator
            recent_arr = np.array(self.recent_rewards)
            reward_var = np.var(recent_arr)
            reward_mean = np.mean(recent_arr)
            
            # Also compute prediction error
            inv = self.A_inv[arm]
            theta = inv @ self.b[arm]
            pred = theta @ context
            err = abs(reward - pred)
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * err
            
            # Combine signals: high variance OR high error -> faster forgetting
            # Normalize variance to [0, 1]
            norm_var = min(reward_var / (1.0 + reward_var), 1.0)
            norm_err = min(err / (1.0 + self.error_ema), 1.0)
            nonstationarity = 0.6 * norm_var + 0.4 * norm_err
            
            # Map to gamma: high non-stationarity -> lower gamma
            self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * nonstationarity
        else:
            self.gamma = 0.99
        
        # Adaptive C: reduce exploration as we learn more
        if self.round_count > 50:
            # Use reward EMA to gauge learning progress
            self.reward_ema = (1 - self.reward_alpha) * self.reward_ema + self.reward_alpha * reward
            # Reduce C as round count increases, but keep some exploration
            progress = min(self.round_count / 500.0, 1.0)
            self.C = self.C_max - (self.C_max - self.C_min) * progress
        else:
            self.C = self.C_max * 0.7
        
        # Update b with forgetting
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update A^{-1} using Sherman-Morrison with forgetting
        V = self.A_inv[arm]
        g = self.gamma
        x = context
        Vx = V @ x
        denom = (1.0 / g) + x @ Vx
        # V_new = (1/g) * V - (1/g^2) * Vx * Vx^T / denom
        self.A_inv[arm] = (1.0 / g) * V - (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        # Symmetrize to avoid numerical drift
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
# EVOLVE-BLOCK-END
