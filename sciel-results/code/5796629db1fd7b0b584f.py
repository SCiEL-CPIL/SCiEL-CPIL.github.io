"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Uses a per-arm ridge regression with an exponential forgetting factor to track
    drifting arm weights. The exploration bonus is adaptive, shrinking as the
    policy gains confidence in its estimates.
    
    Approach:
    - Maintain A[a] = sum of outer(context, context) * decay^age and b[a] = sum of reward*context * decay^age
    - Estimate theta[a] = A[a]^{-1} b[a]
    - Select arm with highest theta[a] @ context + alpha * sqrt(context^T A[a]^{-1} context)
    - Update with forgetting factor to adapt to non-stationarity
    - Uses Sherman-Morrison-Woodbury for efficient inverse updates
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor: high enough to track drift, low enough to retain info
        self.decay = 0.91
        # Exploration parameter
        self.alpha = 0.7
        # Initial round counter for alpha decay
        self.round_count = 0
        # Track reward variance to adapt alpha
        self.reward_std = [1.0] * n_arms
        self.reward_mean = [0.0] * n_arms
        self.reward_count = [0] * n_arms
        self.reward_sum_sq = [0.0] * n_arms

        # Deadline guard
        self.t0 = time.time()
        self.deadline = 145.0  # 5s safety margin

    def _check_deadline(self):
        """Return True if we should stop exploring and just pick greedily."""
        return (time.time() - self.t0) > self.deadline

    def select(self, context):
        if self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64).flatten()
        scores = []
        
        # Adaptive alpha: start moderate, decrease as we learn
        alpha = self.alpha
        if self.round_count > 10:
            # Gradually reduce exploration with a smoother schedule
            alpha = self.alpha * (0.8 + 0.5 * np.exp(-self.round_count / 100.0))
        
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.inv_A[a]
            mean_val = theta @ context
            variance = context @ inv @ context
            # Clip variance to avoid numerical issues
            variance = max(0.0, variance)
            std_val = np.sqrt(variance)
            
            # Scale by reward uncertainty if available
            if self.reward_count[a] > 3:
                reward_var = max(0.0, self.reward_sum_sq[a] / self.reward_count[a] - self.reward_mean[a] ** 2)
                # Use a bounded scale to avoid over-exploration on high-variance arms
                scale = np.sqrt(reward_var) if reward_var > 0 else 1.0
                scale = min(scale, 3.0)  # Cap the scale
            else:
                scale = 1.0
            
            scores.append(mean_val + alpha * scale * std_val)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        arm = int(arm)
        reward = float(reward)
        
        # Track reward statistics for adaptive scaling
        self.reward_count[arm] += 1
        n = self.reward_count[arm]
        # Online update of mean and sum of squares
        self.reward_mean[arm] = (self.reward_mean[arm] * (n - 1) + reward) / n
        self.reward_sum_sq[arm] = self.reward_sum_sq[arm] + reward * reward

        # Apply forgetting
        decay = self.decay
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context

        # Recompute inverse and theta using Sherman-Morrison-Woodbury
        d = self.d
        try:
            # Use SMW for rank-1 update: (decay * A_old + xx^T)^{-1}
            # Let B = decay * A_old, B^{-1} = (1/decay) * A_old^{-1}
            # (B + xx^T)^{-1} = B^{-1} - (B^{-1} xx^T B^{-1}) / (1 + x^T B^{-1} x)
            inv_old = self.inv_A[arm]
            B_inv = (1.0 / decay) * inv_old
            x = context
            xB = B_inv @ x
            denom = 1.0 + x @ xB
            if abs(denom) > 1e-12:
                inv_new = B_inv - np.outer(xB, xB) / denom
            else:
                inv_new = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(d))
            self.inv_A[arm] = inv_new
            self.theta[arm] = inv_new @ self.b[arm]
        except (np.linalg.LinAlgError, Exception):
            # Fallback: compute fresh inverse
            try:
                inv = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(d))
                self.inv_A[arm] = inv
                self.theta[arm] = inv @ self.b[arm]
            except np.linalg.LinAlgError:
                pass

        self.round_count += 1

        # Slowly decay alpha over time
        if self.round_count > 30:
            self.alpha = max(0.12, self.alpha * 0.996)

# EVOLVE-BLOCK-END
