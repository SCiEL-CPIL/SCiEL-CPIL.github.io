"""Non-stationary linear contextual bandit with per-arm adaptive multi-timescale LinUCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Multi-timescale LinUCB with per-arm adaptive forgetting.
    
    Maintains two parallel LinUCB estimators per arm (short/long window) with
    independently adapted forgetting factors. Uses blended UCB for selection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        self.reg = 0.05
        
        # Short-window models (fast forgetting)
        self.A_short = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_short = [0.88 for _ in range(n_arms)]
        
        # Long-window models (slow forgetting)
        self.A_long = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_long = [0.97 for _ in range(n_arms)]
        
        # Per-arm adaptive lambda parameters
        self.lambda_short_base = 0.88
        self.lambda_short_min = 0.75
        self.lambda_short_max = 0.93
        self.lambda_long_base = 0.97
        self.lambda_long_min = 0.90
        self.lambda_long_max = 0.99
        
        # Residual tracking per arm per window
        self.residuals_short = [deque(maxlen=15) for _ in range(n_arms)]
        self.residuals_long = [deque(maxlen=15) for _ in range(n_arms)]
        
        # Global drift tracking
        self.recent_rewards = deque(maxlen=20)
        self.global_mean = 0.0
        self.global_var = 1.0
        
        # Adaptive exploration
        self.alpha_base = 1.2
        self.alpha = 1.2
        self.alpha_min = 0.5
        self.alpha_max = 3.5
        self.alpha_decay_rate = 0.0003
        
        # Blend weight for short vs long window
        self.blend_weight = 0.7
        self.blend_weight_min = 0.5
        self.blend_weight_max = 0.9
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _solve(self, A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_variance(self, A, context):
        try:
            A_inv_context = np.linalg.solve(A, context)
            v = context @ A_inv_context
            return max(v, 0.0)
        except np.linalg.LinAlgError:
            return 1.0

    def _update_lambda(self, arm, residual_sq, is_short):
        """Update per-arm lambda based on recent residuals."""
        if is_short:
            self.residuals_short[arm].append(residual_sq)
            res = self.residuals_short[arm]
        else:
            self.residuals_long[arm].append(residual_sq)
            res = self.residuals_long[arm]
        
        if len(res) < 4:
            return
        
        recent = list(res)
        mean_res = np.mean(recent)
        std_res = np.std(recent)
        
        # MAD-based robust scaling
        mad = np.median(np.abs(np.array(recent) - np.median(recent)))
        scale = max(mad * 1.4826, 0.1)  # Robust scale estimate
        
        error_metric = (mean_res + std_res) / scale
        
        if is_short:
            target = self.lambda_short_base - 0.04 * np.log1p(error_metric)
            target = max(self.lambda_short_min, min(self.lambda_short_max, target))
            self.lambda_short[arm] = 0.9 * self.lambda_short[arm] + 0.1 * target
        else:
            target = self.lambda_long_base - 0.02 * np.log1p(error_metric)
            target = max(self.lambda_long_min, min(self.lambda_long_max, target))
            self.lambda_long[arm] = 0.95 * self.lambda_long[arm] + 0.05 * target

    def _check_drift(self, reward):
        """Detect drift using recent reward statistics."""
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) < 5:
            return False, 0.0
            
        recent_mean = np.mean(self.recent_rewards)
        recent_var = np.var(self.recent_rewards)
        
        self.global_mean = 0.95 * self.global_mean + 0.05 * recent_mean
        self.global_var = 0.95 * self.global_var + 0.05 * recent_var
        
        deviation = abs(recent_mean - self.global_mean)
        threshold = 0.3 * (1.0 + np.sqrt(self.global_var + 1e-6))
        
        is_drift = deviation > threshold
        drift_score = min(1.0, deviation / (threshold + 1e-6))
        
        return is_drift, drift_score

    def _update_alpha(self, is_drift, drift_score):
        """Adapt exploration coefficient."""
        total_obs = sum(self.counts)
        
        # Time-based decay
        time_decay = 1.0 - 0.3 * (1.0 - np.exp(-self.t / 100.0))
        
        if is_drift:
            alpha_target = self.alpha_base * time_decay + 0.5 * drift_score
        else:
            alpha_target = self.alpha_base * time_decay
        
        # Slow observation-based decay
        alpha_target = max(self.alpha_min, alpha_target - self.alpha_decay_rate * total_obs)
        
        self.alpha = 0.85 * self.alpha + 0.15 * alpha_target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _select_arm(self, context):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adapt blend weight
        if hasattr(self, 'drift_score_recent') and self.drift_score_recent > 0.5:
            target_blend = self.blend_weight_max
        else:
            target_blend = self.blend_weight
        
        self.blend_weight = 0.9 * self.blend_weight + 0.1 * target_blend
        self.blend_weight = max(self.blend_weight_min, min(self.blend_weight_max, self.blend_weight))

        scores = []
        for a in range(self.n):
            theta_s = self._solve(self.A_short[a], self.b_short[a])
            mu_s = theta_s @ context
            v_s = self._get_variance(self.A_short[a], context)
            
            theta_l = self._solve(self.A_long[a], self.b_long[a])
            mu_l = theta_l @ context
            v_l = self._get_variance(self.A_long[a], context)
            
            disagreement = abs(mu_s - mu_l)
            disagreement_bonus = 0.3 * np.sqrt(disagreement + 1e-6)
            
            ucb_s = mu_s + self.alpha * np.sqrt(v_s) + disagreement_bonus
            ucb_l = mu_l + self.alpha * 0.7 * np.sqrt(v_l)
            
            score = self.blend_weight * ucb_s + (1 - self.blend_weight) * ucb_l
            
            if self.counts[a] < 3:
                score += 0.4 / (1 + self.counts[a])
            
            scores.append(score)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        if time.time() > self.deadline:
            return 0
        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        theta_s = self._solve(self.A_short[arm], self.b_short[arm])
        mu_s = theta_s @ context
        residual_s = reward - mu_s
        residual_sq_s = residual_s ** 2
        
        theta_l = self._solve(self.A_long[arm], self.b_long[arm])
        mu_l = theta_l @ context
        residual_l = reward - mu_l
        residual_sq_l = residual_l ** 2
        
        self._update_lambda(arm, residual_sq_s, True)
        self._update_lambda(arm, residual_sq_l, False)
        
        is_drift, drift_score = self._check_drift(reward)
        self.drift_score_recent = drift_score
        self._update_alpha(is_drift, drift_score)
        
        if is_drift and drift_score > 0.6:
            self.lambda_short[arm] = max(self.lambda_short_min, self.lambda_short[arm] * 0.96)
            self.lambda_long[arm] = max(self.lambda_long_min, self.lambda_long[arm] * 0.98)
        
        if not is_drift:
            self.lambda_short[arm] = 0.995 * self.lambda_short[arm] + 0.005 * self.lambda_short_base
            self.lambda_long[arm] = 0.995 * self.lambda_long[arm] + 0.005 * self.lambda_long_base
        
        lam_s = self.lambda_short[arm]
        self.A_short[arm] = lam_s * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = lam_s * self.b_short[arm] + reward * context
        
        lam_l = self.lambda_long[arm]
        self.A_long[arm] = lam_l * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_l * self.b_long[arm] + reward * context
        
        if self.t % 100 == 0:
            for a in range(self.n):
                self.A_short[a] = self.A_short[a] + self.reg * 0.1 * np.eye(self.d)
                self.A_long[a] = self.A_long[a] + self.reg * 0.1 * np.eye(self.d)
        
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
