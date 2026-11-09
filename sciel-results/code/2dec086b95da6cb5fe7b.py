"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Refined Adaptive LinUCB with robust drift detection and smoothed forgetting.
    
    Key improvements over previous versions:
    1. Statistically robust drift detection using standard deviation when possible.
    2. Smoothed lambda updates to prevent oscillation in memory length.
    3. Higher initial regularization for stability.
    4. Optimized linear algebra calls.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Higher initial regularization to reduce early instability
        self.A = [np.eye(dim) * 0.5 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        self.lambdas = [0.98 for _ in range(n_arms)]
        self.lambda_min = 0.80
        self.lambda_max = 0.995
        self.lambda_base = 0.98
        
        # Tracking for error and drift
        # Store recent squared errors
        self.error_hist = [deque(maxlen=20) for _ in range(n_arms)]
        # Store recent rewards for drift detection
        self.reward_hist = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration parameters
        self.base_alpha = 1.2  # Slightly higher base to ensure early exploration
        self.t = 0
        
        # Counters for adaptive alpha
        self.count = [0] * n_arms
        self.total_reward = [0.0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin before 150s kill

    def _solve(self, A, b):
        """Solve A x = b efficiently with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_adaptive_params(self, arm, residual_sq, reward):
        """
        Update the forgetting factor for the specified arm based on 
        prediction error and robust drift detection.
        """
        # 1. Update histories
        self.error_hist[arm].append(residual_sq)
        self.reward_hist[arm].append(reward)
        
        # 2. Detect Drift
        # We use a robust z-score approach if we have enough data, otherwise a heuristic
        is_drift = False
        hist = self.reward_hist[arm]
        
        if len(hist) >= 5:
            recent = list(hist)
            # Use the last 5 rewards for local statistics
            window = recent[-5:]
            mean_r = np.mean(window[:-1])
            std_r = np.std(window[:-1])
            
            # If std is very small, use a minimum threshold to avoid division by zero
            # or excessive sensitivity to tiny noise
            min_std = 0.1
            std_r = max(std_r, min_std)
            
            # Z-score of the latest reward relative to the previous 4
            z_score = abs(reward - mean_r) / std_r
            
            # If z-score is high, it's likely a drift
            if z_score > 2.5:
                is_drift = True
        elif len(hist) >= 3:
            # Fallback heuristic for short history
            prev_avg = np.mean(list(hist)[:-1])
            # Heuristic: deviation > 0.5 * (1 + abs(mean))
            if abs(reward - prev_avg) > 0.5 * (1.0 + abs(prev_avg)):
                is_drift = True

        # 3. Determine new Lambda
        # Base adjustment based on average prediction error
        # Higher error -> lower lambda (more forgetting)
        avg_err_sq = np.mean(self.error_hist[arm]) if self.error_hist[arm] else 0.0
        
        # Sensitivity factor: how much error affects lambda
        # lambda = base / (1 + k * err)
        # We tune k to be moderate
        new_lambda = self.lambda_base / (1.0 + 0.5 * avg_err_sq)
        
        # If drift is detected, force a more aggressive forgetting
        # We don't set it to minimum immediately, but push it down significantly
        if is_drift:
            new_lambda = min(new_lambda, 0.90)
            
        # Clamp to valid range
        target_lambda = max(self.lambda_min, min(self.lambda_max, new_lambda))
        
        # Smooth the lambda update to avoid oscillation
        # Move current lambda towards target lambda
        self.lambdas[arm] = 0.8 * self.lambdas[arm] + 0.2 * target_lambda

    def _select_arm(self, context):
        """Select arm using LinUCB with per-arm adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adaptive global alpha: decay over time to reduce exploration
        # Slower decay to maintain some exploration for non-stationary shifts
        alpha_global = self.base_alpha * max(0.6, 1.0 - 0.2 * min(1.0, self.t / 500.0))

        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Solve for theta
            theta = self._solve(A, b)
            mu = theta @ context
            
            # Compute variance term: context^T A^{-1} context
            try:
                A_inv_context = np.linalg.solve(A, context)
                v = context @ A_inv_context
            except np.linalg.LinAlgError:
                v = 1.0
            
            # Per-arm adaptive alpha: boost exploration for rarely pulled arms
            # This ensures all arms get explored early on
            alpha_arm = alpha_global * (1.0 + 0.5 / (1.0 + self.count[a]))
            
            scores.append(mu + alpha_arm * np.sqrt(max(v, 0.0)))

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return arm with highest historical average reward
            if any(self.count[a] > 0 for a in range(self.n)):
                avgs = [self.total_reward[a] / max(self.count[a], 1) for a in range(self.n)]
                return int(np.argmax(avgs))
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # 1. Compute prediction to get residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update adaptive forgetting factor
        self._update_adaptive_params(arm, residual_sq, reward)

        # 3. Update A and b for the pulled arm using the new lambda
        lam = self.lambdas[arm]
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        # 4. Update counters for adaptive alpha
        self.count[arm] += 1
        # Decay total reward to reflect recent performance more heavily
        self.total_reward[arm] = lam * self.total_reward[arm] + reward


# EVOLVE-BLOCK-END
