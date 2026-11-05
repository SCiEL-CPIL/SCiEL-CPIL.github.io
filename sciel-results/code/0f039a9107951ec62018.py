# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    Uses an exponential forgetting factor with adaptive gamma and exploration.
    Stores inverse covariance for O(d^2) updates instead of O(d^3) inversions.
    Includes a deadline guard to return a safe arm if time runs low.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 140s deadline (20s safety margin)
        
        # Inverse covariance matrices V_a = A_a^{-1}
        self.V = [np.eye(dim) for _ in range(n_arms)]
        # Reward-weighted feature vectors b_a
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Counts for adaptive gamma
        self.counts = [0] * n_arms
        self.total_steps = 0
        
        # Adaptive parameters
        self.gamma = 0.96  # Base forgetting factor
        self.gamma_min = 0.90
        self.gamma_max = 0.995
        self.C = 1.5  # Exploration parameter
        self.C_min = 0.5
        self.C_max = 3.0
        
        # For gamma adaptation
        self.reward_history = []
        self.gamma_adapt_window = 50
        self.prev_mean_reward = None
        self.reward_variance = 0.0

    def _check_deadline(self):
        """Return True if we are past the deadline and should use a safe fallback."""
        return time.time() > self.deadline

    def _select_safe_arm(self, context):
        """Fallback selection when deadline is near: pick arm with highest current estimate."""
        best_arm = 0
        best_score = -np.inf
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            score = theta @ context
            if score > best_score:
                best_score = score
                best_arm = a
        return int(best_arm)

    def _adapt_parameters(self, reward):
        """Adapt gamma and C based on recent reward statistics."""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.gamma_adapt_window:
            self.reward_history.pop(0)
        
        if len(self.reward_history) >= 10:
            mean_reward = np.mean(self.reward_history)
            if self.prev_mean_reward is not None:
                # If mean reward is dropping, increase forgetting (lower gamma)
                # If mean reward is stable or rising, can increase gamma (slower forgetting)
                delta_mean = mean_reward - self.prev_mean_reward
                if delta_mean < -0.1:
                    # Regime change detected, increase forgetting
                    self.gamma = max(self.gamma_min, self.gamma * 0.98)
                    # Increase exploration
                    self.C = min(self.C_max, self.C * 1.05)
                elif delta_mean > 0.1:
                    # Improving, can slow forgetting
                    self.gamma = min(self.gamma_max, self.gamma * 1.02)
                    # Decrease exploration
                    self.C = max(self.C_min, self.C * 0.98)
                else:
                    # Stable, nudge gamma toward base
                    self.gamma = 0.96 + 0.04 * (self.gamma - 0.96)
                    self.C = 1.5 + 0.5 * (self.C - 1.5)
            self.prev_mean_reward = mean_reward

    def select(self, context):
        """Select arm using UCB with adaptive exploration."""
        if self._check_deadline():
            return self._select_safe_arm(context)
        
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            # UCB score: mean + C * sqrt(context^T V context)
            uncertainty = np.sqrt(context @ self.V[a] @ context)
            scores[a] = theta @ context + self.C * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update model with new observation using inverse matrix update."""
        if self._check_deadline():
            # Still update but skip adaptation to save time
            context = np.asarray(context, dtype=np.float64)
            self._update_model(arm, context, reward)
            self.total_steps += 1
            return
        
        context = np.asarray(context, dtype=np.float64)
        self._update_model(arm, context, reward)
        self.total_steps += 1
        self._adapt_parameters(reward)

    def _update_model(self, arm, context, reward):
        """Core model update with forgetting factor."""
        g = self.gamma
        # Forget old information
        self.V[arm] = self.V[arm] / g
        self.b[arm] = g * self.b[arm]
        
        # Add new observation
        outer = np.outer(context, context)
        # V_new = (V_old + outer)^{-1} using Sherman-Morrison
        # (A + xy^T)^{-1} = A^{-1} - A^{-1}xy^T(A + y^TA^{-1})^{-1}
        # Here V = A^{-1}, so new V = (V^{-1} + outer)^{-1}
        # Using inverse update: V_new = V - V @ outer @ V / (1 + context^T @ V @ context)
        V = self.V[arm]
        Vx = V @ context
        denom = 1.0 + context @ Vx
        self.V[arm] = V - np.outer(Vx, Vx) / denom
        self.b[arm] = self.b[arm] + reward * context
        self.counts[arm] += 1
# EVOLVE-BLOCK-END
