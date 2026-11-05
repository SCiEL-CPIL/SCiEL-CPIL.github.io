"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor and maintained inverse."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Maintain A^{-1} directly with ridge for stability
        self.ridge = 0.15
        self.A_inv = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Aggressive forgetting for non-stationarity
        self.gamma = 0.95
        self.gamma_min = 0.75
        self.gamma_max = 0.98
        self.error_ema = 0.0
        self.error_alpha = 0.25
        
        # Adaptive exploration parameter - start lower
        self.C = 1.0
        self.C_min = 0.2
        self.C_max = 1.5
        self.reward_ema = 0.0
        self.reward_alpha = 0.15
        
        # Track recent rewards for non-stationarity detection
        self.recent_rewards = []
        self.recent_reward_window = 30
        
        self.round_count = 0
        
        # Tighter deadline guard
        self._start_time = time.time()
        self._deadline = 135.0  # seconds - safety margin
        self._last_update_time = time.time()

    def select(self, context):
        # Deadline guard: if we're getting close to the limit, be conservative
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline - 3.0:
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
        if time.time() - self._start_time > self._deadline - 1.0:
            return
        
        # Track recent rewards for non-stationarity detection
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_reward_window:
            self.recent_rewards.pop(0)
        
        # Adaptive gamma: detect non-stationarity via prediction error
        if self.round_count > 10 and len(self.recent_rewards) >= 5:
            # Compute prediction error
            inv = self.A_inv[arm]
            theta = inv @ self.b[arm]
            pred = theta @ context
            err = abs(reward - pred)
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * err
            
            # High error -> faster forgetting (lower gamma)
            # Normalize error to [0, 1]
            norm_err = min(err / (1.0 + self.error_ema), 1.0)
            
            # Map to gamma: high error -> lower gamma
            self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * norm_err
        else:
            self.gamma = 0.95
        
        # Adaptive C: reduce exploration quickly after initial learning
        if self.round_count > 30:
            # Reduce C as round count increases, but keep some exploration
            progress = min(self.round_count / 300.0, 1.0)
            self.C = self.C_max - (self.C_max - self.C_min) * progress
        else:
            self.C = self.C_max * 0.8
        
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
