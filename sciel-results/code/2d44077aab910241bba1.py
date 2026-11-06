import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear contextual bandits.
    Combines:
    - Structural idea: Per-arm adaptive forgetting factor (lambda) based on local MSE.
    - Refinement machinery: Global adaptive alpha (exploration) based on global MSE,
      and a strict deadline guard for robustness.
    - Cached inverses for speed.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]  # Cached inverses
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.88
        self.lambda_max = 0.998
        self.error_windows = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration
        self.alpha_base = 1.0
        self.alpha_min = 0.3
        self.alpha_max = 2.5
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=50)
        
        # Short-term reward tracking for regime change detection
        self.recent_rewards = deque(maxlen=15)
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _update_inv_woodbury(self, arm, context, lam):
        """
        Update inverse using Woodbury identity for A_new = lam * A_old + x x^T.
        A_new^{-1} = (lam A_old + x x^T)^{-1}
        Let M = lam A_old. M^{-1} = (1/lam) A_old^{-1}.
        (M + x x^T)^{-1} = M^{-1} - (M^{-1} x x^T M^{-1}) / (1 + x^T M^{-1} x)
        """
        inv_old = self.inv_A[arm]
        # M^{-1} = (1/lam) * inv_old
        inv_M = inv_old / lam
        
        # Compute v = M^{-1} x
        v = inv_M @ context
        
        # Compute denominator: 1 + x^T v
        denom = 1.0 + context @ v
        
        # Update inverse
        # inv_new = inv_M - (v v^T) / denom
        self.inv_A[arm] = inv_M - np.outer(v, v) / denom

    def _get_lambda(self, arm):
        """Determine forgetting factor based on recent errors for this arm."""
        if len(self.error_windows[arm]) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows[arm])
        mse = np.mean(errors**2)
        
        # Use a sharper transition: smaller scale means faster response to error
        # Tuned scale for better responsiveness
        scale = 0.02
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance."""
        if len(self.global_errors) < 5:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        # Also consider trend: are errors increasing or decreasing?
        if len(errors) >= 10:
            half = len(errors) // 2
            recent_half_mse = np.mean(errors[half:]**2)
            older_half_mse = np.mean(errors[:half]**2)
            trend = recent_half_mse - older_half_mse  # positive = getting worse
        else:
            trend = 0.0
        
        # Base target from MSE
        # Adjusted scale for alpha adaptation
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.1))
        
        # Adjust for trend: if errors are increasing, boost exploration more
        if trend > 0.01:
            target_alpha *= 1.15
        elif trend < -0.01:
            target_alpha *= 0.95
        
        # Smoothly move towards target
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_reward_stats(self, reward):
        """Update running statistics of rewards for regime detection."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) >= 5:
            rewards = np.array(self.recent_rewards)
            self.reward_mean = np.mean(rewards)
            self.reward_std = max(np.std(rewards), 0.1)

    def select(self, context):
        if not self._check_deadline():
            # If we are near the deadline, just pick the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta = self.inv_A[a] @ self.b[a]
                    mu = theta @ context
                except:
                    mu = 0.0
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            mu = theta @ context
            var = context @ self.inv_A[a] @ context
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Calculate residual using the model *before* this update
        theta_old = self.inv_A[arm] @ self.b[arm]
        pred = theta_old @ context
        residual = reward - pred
        
        # Store residual
        self.error_windows[arm].append(residual)
        self.global_errors.append(residual)
        
        # Update reward statistics
        self._update_reward_stats(reward)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda(arm)
        
        # 3. Update matrices
        # We need A for potential future checks or if Woodbury fails, but primarily for consistency if we ever need it.
        # However, for the Woodbury update, we only need inv_A.
        # We still update A and b for the sake of the "state" being correct if we were to restart or use A elsewhere.
        # But wait, if we only maintain inv_A, we don't strictly need A unless we need to check condition number or something.
        # The current code uses A in _recompute_inv. If we replace _recompute_inv with Woodbury, we don't need A for the inverse update.
        # Do we need A for anything else? No, only inv_A is used in select and update.
        # So we can skip updating A? 
        # If we skip updating A, A becomes stale. But we don't use A. 
        # However, if we want to be safe, we can update A. It's O(d^2).
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # Update inverse using Woodbury
        self._update_inv_woodbury(arm, context, lam)
        
        # 4. Update alpha
        self._update_alpha()
