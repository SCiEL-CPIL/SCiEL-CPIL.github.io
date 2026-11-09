"""Non-stationary linear contextual bandit with multi-timescale LinUCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Multi-timescale LinUCB for non-stationary linear contextual bandits.
    
    Maintains two parallel LinUCB estimators per arm:
    - Short-window: rapid adaptation to recent weight shifts (lower lambda)
    - Long-window: stable baseline estimation (higher lambda)
    
    Arm selection uses a blend of short-window UCB (for adaptation) and
    long-window UCB (for stability). Adaptive alpha increases during drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Regularization
        self.reg = 0.1
        
        # Short-window models (fast forgetting)
        self.A_short = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_short = 0.90  # Faster forgetting
        
        # Long-window models (slow forgetting)
        self.A_long = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_long = 0.98  # Slower forgetting
        
        # Adaptive parameters
        self.lambda_short_base = 0.90
        self.lambda_short_min = 0.80
        self.lambda_short_max = 0.95
        self.lambda_long_base = 0.98
        self.lambda_long_min = 0.92
        self.lambda_long_max = 0.995
        
        # Residual tracking for adaptive lambda
        self.residuals_short = [np.zeros(20) for _ in range(n_arms)]
        self.residuals_long = [np.zeros(20) for _ in range(n_arms)]
        self.residual_idx = [0 for _ in range(n_arms)]
        
        # Global drift detection
        self.recent_rewards = []
        self.reward_window = 15
        self.reward_history = []
        self.global_mean = 0.0
        self.global_var = 1.0
        
        # Adaptive exploration
        self.alpha_base = 1.0
        self.alpha = 1.0
        self.alpha_min = 0.6
        self.alpha_max = 3.0
        
        # Blend weight for short vs long window
        self.blend_weight = 0.6  # Weight for short-window in selection
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _solve(self, A, b):
        """Solve A x = b with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_theta(self, A, b):
        """Get theta estimate from A and b."""
        return self._solve(A, b)

    def _get_variance(self, A, context):
        """Compute context^T A^{-1} context."""
        try:
            A_inv_context = np.linalg.solve(A, context)
            v = context @ A_inv_context
            return max(v, 0.0)
        except np.linalg.LinAlgError:
            return 1.0

    def _update_adaptive_lambda(self, arm, residual_sq, is_short):
        """Update lambda based on recent residuals."""
        idx = self.residual_idx[arm]
        if is_short:
            self.residuals_short[arm][idx] = residual_sq
        else:
            self.residuals_long[arm][idx] = residual_sq
        self.residual_idx[arm] = (idx + 1) % 20
        
        # Need at least 5 samples to adapt
        if self.t < 5:
            return
        
        if is_short:
            res = self.residuals_short[arm]
            mean_res = np.mean(res)
            std_res = np.std(res)
            error_metric = mean_res + std_res
            
            # Lower lambda when error is high (faster forgetting)
            target = self.lambda_short_base - 0.03 * np.log1p(error_metric)
            target = max(self.lambda_short_min, min(self.lambda_short_max, target))
            
            # Smoothly adapt
            self.lambda_short = 0.9 * self.lambda_short + 0.1 * target
        else:
            res = self.residuals_long[arm]
            mean_res = np.mean(res)
            std_res = np.std(res)
            error_metric = mean_res + std_res
            
            target = self.lambda_long_base - 0.02 * np.log1p(error_metric)
            target = max(self.lambda_long_min, min(self.lambda_long_max, target))
            
            self.lambda_long = 0.95 * self.lambda_long + 0.05 * target

    def _check_drift(self, reward):
        """Detect drift using recent reward statistics."""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 5:
            return False, 0.0
            
        recent_mean = np.mean(self.reward_history)
        recent_var = np.var(self.reward_history)
        
        # Update global stats
        self.global_mean = 0.95 * self.global_mean + 0.05 * recent_mean
        self.global_var = 0.95 * self.global_var + 0.05 * recent_var
        
        # Drift score: how much recent mean deviates from global
        deviation = abs(recent_mean - self.global_mean)
        threshold = 0.3 * (1.0 + np.sqrt(self.global_var + 1e-6))
        
        is_drift = deviation > threshold
        drift_score = min(1.0, deviation / (threshold + 1e-6))
        
        return is_drift, drift_score

    def _update_alpha(self, is_drift, drift_score):
        """Adapt exploration coefficient."""
        if is_drift:
            # Increase alpha during drift
            self.alpha = min(self.alpha_max, self.alpha + 0.1 * drift_score)
        else:
            # Decay back to base
            self.alpha = max(self.alpha_base, self.alpha * 0.995)
        
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _select_arm(self, context):
        """Select arm using blended multi-timescale UCB."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        for a in range(self.n):
            # Short-window estimate
            theta_s = self._get_theta(self.A_short[a], self.b_short[a])
            mu_s = theta_s @ context
            v_s = self._get_variance(self.A_short[a], context)
            ucb_s = mu_s + self.alpha * np.sqrt(v_s)
            
            # Long-window estimate
            theta_l = self._get_theta(self.A_long[a], self.b_long[a])
            mu_l = theta_l @ context
            v_l = self._get_variance(self.A_long[a], context)
            ucb_l = mu_l + self.alpha * 0.8 * np.sqrt(v_l)  # Slightly less exploration on long
            
            # Blend: short window for adaptation, long for stability
            score = self.blend_weight * ucb_s + (1 - self.blend_weight) * ucb_l
            
            # Boost for arms with very few pulls
            if self.counts[a] < 3:
                score += 0.5 / (1 + self.counts[a])
            
            scores.append(score)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute residuals for adaptive lambda
        theta_s = self._get_theta(self.A_short[arm], self.b_short[arm])
        mu_s = theta_s @ context
        residual_s = reward - mu_s
        residual_sq_s = residual_s ** 2
        
        theta_l = self._get_theta(self.A_long[arm], self.b_long[arm])
        mu_l = theta_l @ context
        residual_l = reward - mu_l
        residual_sq_l = residual_l ** 2
        
        # Update adaptive lambdas
        self._update_adaptive_lambda(arm, residual_sq_s, True)
        self._update_adaptive_lambda(arm, residual_sq_l, False)
        
        # Check drift and update alpha
        is_drift, drift_score = self._check_drift(reward)
        self._update_alpha(is_drift, drift_score)
        
        # If significant drift, temporarily increase forgetting
        if is_drift and drift_score > 0.8:
            self.lambda_short = max(self.lambda_short_min, self.lambda_short * 0.95)
            self.lambda_long = max(self.lambda_long_min, self.lambda_long * 0.97)
        
        # Update short-window model
        lam_s = self.lambda_short
        self.A_short[arm] = lam_s * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = lam_s * self.b_short[arm] + reward * context
        
        # Update long-window model
        lam_l = self.lambda_long
        self.A_long[arm] = lam_l * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_l * self.b_long[arm] + reward * context
        
        # Update count
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
