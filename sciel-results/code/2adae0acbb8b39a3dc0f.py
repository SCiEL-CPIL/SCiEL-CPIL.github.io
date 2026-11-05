"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Bandit with:
    - Per-arm exponential forgetting (discount factor gamma)
    - Adaptive gamma based on prediction error (detects regime changes)
    - UCB exploration with adaptive confidence parameter
    - Sherman-Morrison update for A^{-1} with discounting
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        
        # Ridge regularization for numerical stability
        self.ridge = 0.1
        
        # Initialize A_inv as (ridge * I)^{-1} = I / ridge
        self.A_inv = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Discount factor for non-stationarity
        self.gamma = 0.95
        self.gamma_min = 0.80
        self.gamma_max = 0.98
        self.error_ema = 0.0
        self.error_alpha = 0.3
        self.reward_ema = 0.0
        self.reward_alpha = 0.2
        
        # Exploration parameter
        self.C = 1.0
        self.C_min = 0.3
        self.C_max = 2.0
        
        # Track recent prediction errors for adaptive gamma
        self.recent_errors = []
        self.error_window = 20
        
        self.round_count = 0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 135.0  # safety margin from 150s limit
        self._last_update_time = time.time()
        
        # Best arm tracking for fallback
        self.best_arm = 0
        self.arm_rewards = [0.0 for _ in range(n_arms)]
        self.arm_counts = [0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard: if we're getting close to the limit, be conservative
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline - 3.0:
            # Return the arm with highest mean reward estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                if self.arm_counts[a] > 0:
                    score = self.arm_rewards[a] / self.arm_counts[a]
                else:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            # UCB: mean + exploration bonus
            mean = theta @ context
            variance = context @ inv @ context
            bonus = self.C * np.sqrt(max(variance, 0.0))
            scores.append(mean + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline before doing expensive work
        if time.time() - self._start_time > self._deadline - 1.0:
            return
        
        # Update arm statistics for fallback
        self.arm_rewards[arm] += reward
        self.arm_counts[arm] += 1
        
        # Compute prediction error for adaptive gamma
        inv = self.A_inv[arm]
        theta = inv @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        
        self.recent_errors.append(err)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)
        
        # Adaptive gamma: high error -> faster forgetting (lower gamma)
        if self.round_count > 5 and len(self.recent_errors) >= 5:
            avg_err = np.mean(self.recent_errors)
            # Normalize error: map to [0, 1]
            # If avg_err is small, gamma stays high (slow forgetting)
            # If avg_err is large, gamma drops (fast forgetting)
            norm_err = min(avg_err / (0.5 + self.error_ema), 1.0)
            self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * avg_err
            self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * norm_err
        else:
            self.gamma = 0.95
        
        # Adaptive C: reduce exploration as we learn
        if self.round_count > 20:
            progress = min(self.round_count / 200.0, 1.0)
            self.C = self.C_max - (self.C_max - self.C_min) * progress
        else:
            self.C = self.C_max * 0.7
        
        # Update b with forgetting
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Update A^{-1} using Sherman-Morrison with forgetting
        # A_new = gamma * A_old + x x^T
        # A_new^{-1} = (1/gamma) * (A_old^{-1} - (1/gamma) * A_old^{-1} x x^T A_old^{-1} / (1 + (1/gamma) x^T A_old^{-1} x))
        V = self.A_inv[arm]
        g = self.gamma
        x = context
        Vx = V @ x
        xVx = x @ Vx
        
        # Sherman-Morrison with discounting
        # V_new = (1/g) * V - (1/g^2) * outer(Vx, Vx) / (1/g + x^T V x)
        denom = (1.0 / g) + xVx
        
        if denom < 1e-10:
            # Numerical issue, keep old inverse
            return
        
        self.A_inv[arm] = (1.0 / g) * V - (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        # Symmetrize to avoid numerical drift
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
# EVOLVE-BLOCK-END
