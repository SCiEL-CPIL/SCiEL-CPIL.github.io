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
        
        # Regularization
        self.reg = 0.05
        
        # Short-window models (fast forgetting) - per-arm
        self.A_short = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_short = [0.88] * n_arms  # Per-arm lambda
        
        # Long-window models (slow forgetting) - per-arm
        self.A_long = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_long = [0.97] * n_arms  # Per-arm lambda
        
        # Adaptive parameters
        self.lambda_short_base = 0.88
        self.lambda_short_min = 0.72
        self.lambda_short_max = 0.94
        self.lambda_long_base = 0.97
        self.lambda_long_min = 0.88
        self.lambda_long_max = 0.995
        
        # Per-arm residual tracking for adaptive lambda
        self.residuals_short = [deque(maxlen=25) for _ in range(n_arms)]
        self.residuals_long = [deque(maxlen=25) for _ in range(n_arms)]
        
        # Per-arm disagreement tracking
        self.disagreement = [0.0] * n_arms
        self.disagreement_smooth = [0.0] * n_arms
        
        # Global drift detection
        self.reward_window = 20
        self.reward_history = deque(maxlen=self.reward_window)
        self.global_mean = 0.0
        self.global_var = 1.0
        
        # Adaptive exploration
        self.alpha_base = 1.3
        self.alpha = 1.3
        self.alpha_min = 0.4
        self.alpha_max = 4.0
        
        # Blend weight for short vs long window
        self.blend_weight = 0.65
        self.blend_weight_min = 0.45
        self.blend_weight_max = 0.9
        
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

    def _update_adaptive_lambda(self, arm, residual, is_short):
        """Update per-arm lambda based on recent residuals using MAD-based robust scaling."""
        if is_short:
            self.residuals_short[arm].append(abs(residual))
        else:
            self.residuals_long[arm].append(abs(residual))
        
        # Need at least 8 samples to adapt
        if self.t < 8:
            return
        
        if is_short:
            res = np.array(self.residuals_short[arm])
            if len(res) < 8:
                return
            # MAD-based robust scale
            med = np.median(res)
            mad = np.median(np.abs(res - med))
            # Robust z-score of current residual
            robust_scale = 1.4826 * mad if mad > 1e-6 else np.std(res) + 1e-6
            z_score = abs(residual) / robust_scale if robust_scale > 1e-6 else 0.0
            
            # Higher z-score means more anomalous -> faster forgetting
            # Use smooth sigmoid-like mapping
            drift_signal = min(1.0, z_score / 3.0)
            
            # Target lambda: lower when drift detected
            target = self.lambda_short_base - 0.15 * drift_signal
            target = max(self.lambda_short_min, min(self.lambda_short_max, target))
            
            # Smooth adaptation with faster response to drift
            rate = 0.15 + 0.25 * drift_signal
            self.lambda_short[arm] = (1 - rate) * self.lambda_short[arm] + rate * target
        else:
            res = np.array(self.residuals_long[arm])
            if len(res) < 8:
                return
            med = np.median(res)
            mad = np.median(np.abs(res - med))
            robust_scale = 1.4826 * mad if mad > 1e-6 else np.std(res) + 1e-6
            z_score = abs(residual) / robust_scale if robust_scale > 1e-6 else 0.0
            
            drift_signal = min(1.0, z_score / 4.0)
            
            target = self.lambda_long_base - 0.08 * drift_signal
            target = max(self.lambda_long_min, min(self.lambda_long_max, target))
            
            rate = 0.08 + 0.12 * drift_signal
            self.lambda_long[arm] = (1 - rate) * self.lambda_long[arm] + rate * target

    def _check_drift(self, reward):
        """Detect drift using recent reward statistics with robust scaling."""
        self.reward_history.append(reward)
        
        if len(self.reward_history) < 8:
            return False, 0.0
            
        recent = np.array(self.reward_history)
        recent_mean = np.mean(recent)
        
        # Update global stats with robust approach
        self.global_mean = 0.93 * self.global_mean + 0.07 * recent_mean
        
        # Use MAD for robust variance estimate
        med = np.median(recent)
        mad = np.median(np.abs(recent - med))
        robust_std = 1.4826 * mad if mad > 1e-6 else np.std(recent) + 1e-6
        
        # Drift score based on deviation from running mean, scaled by robust std
        deviation = abs(recent_mean - self.global_mean)
        threshold = 0.5 * robust_std + 0.2
        
        # Smooth drift score using softplus-like function
        raw_score = deviation / (threshold + 1e-6)
        drift_score = min(1.0, raw_score / (1.0 + raw_score))
        
        is_drift = drift_score > 0.35
        
        return is_drift, drift_score

    def _update_alpha(self, is_drift, drift_score):
        """Adapt exploration coefficient using drift, model disagreement, and time-based decay."""
        # Store recent drift score for blend adaptation
        self.drift_score_recent = drift_score
        
        # Time-based decay: smooth exponential decay
        time_decay = 1.0 - 0.35 * (1.0 - np.exp(-self.t / 80.0))
        
        # Base alpha adjustment from drift with smooth transition
        drift_boost = 0.6 * drift_score
        alpha_target = self.alpha_base * time_decay + drift_boost
        
        # Additional boost from per-arm disagreement (smoothed)
        if self.t > 10:
            # Use smoothed max disagreement across arms
            smooth_disagreement = max(self.disagreement_smooth)
            disagreement_boost = min(0.4, smooth_disagreement * 0.6)
            alpha_target += disagreement_boost
        
        # Smooth adaptation with faster response during drift
        rate = 0.12 + 0.18 * drift_score
        self.alpha = (1 - rate) * self.alpha + rate * alpha_target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _select_arm(self, context):
        """Select arm using blended multi-timescale UCB with per-arm disagreement."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adapt blend weight based on drift detection
        drift_score = getattr(self, 'drift_score_recent', 0.0)
        if drift_score > 0.4:
            # During drift, prioritize short-window more
            target_blend = self.blend_weight_max
        else:
            target_blend = 0.6 + 0.3 * (1 - drift_score)
        
        self.blend_weight = 0.88 * self.blend_weight + 0.12 * target_blend
        self.blend_weight = max(self.blend_weight_min, min(self.blend_weight_max, self.blend_weight))

        scores = []
        max_disagreement = 0.0
        for a in range(self.n):
            # Short-window estimate
            theta_s = self._get_theta(self.A_short[a], self.b_short[a])
            mu_s = theta_s @ context
            v_s = self._get_variance(self.A_short[a], context)
            
            # Long-window estimate
            theta_l = self._get_theta(self.A_long[a], self.b_long[a])
            mu_l = theta_l @ context
            v_l = self._get_variance(self.A_long[a], context)
            
            # Disagreement between models indicates recent drift
            disagreement = abs(mu_s - mu_l)
            max_disagreement = max(max_disagreement, disagreement)
            
            # Smooth per-arm disagreement
            self.disagreement_smooth[a] = 0.85 * self.disagreement_smooth[a] + 0.15 * disagreement
            
            # UCB with disagreement-augmented exploration
            # Use soft disagreement bonus (smooth, bounded)
            disagreement_bonus = 0.25 * np.log1p(disagreement)
            
            # Per-arm adaptive alpha boost based on disagreement
            arm_alpha_boost = 1.0 + 0.3 * min(1.0, self.disagreement_smooth[a])
            
            ucb_s = mu_s + self.alpha * arm_alpha_boost * np.sqrt(v_s) + disagreement_bonus
            ucb_l = mu_l + self.alpha * 0.65 * np.sqrt(v_l)
            
            # Blend: short window for adaptation, long for stability
            score = self.blend_weight * ucb_s + (1 - self.blend_weight) * ucb_l
            
            # Boost for arms with very few pulls (gentle)
            if self.counts[a] < 5:
                score += 0.3 / (1 + self.counts[a])
            
            scores.append(score)

        # Store max disagreement for alpha adaptation
        self.max_disagreement = max_disagreement

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute residuals for adaptive lambda (use raw residual, not squared)
        theta_s = self._get_theta(self.A_short[arm], self.b_short[arm])
        mu_s = theta_s @ context
        residual_s = reward - mu_s
        
        theta_l = self._get_theta(self.A_long[arm], self.b_long[arm])
        mu_l = theta_l @ context
        residual_l = reward - mu_l
        
        # Update per-arm adaptive lambdas
        self._update_adaptive_lambda(arm, residual_s, True)
        self._update_adaptive_lambda(arm, residual_l, False)
        
        # Check drift and update alpha
        is_drift, drift_score = self._check_drift(reward)
        self._update_alpha(is_drift, drift_score)
        
        # Gradual recovery of per-arm lambdas toward base values when no drift
        if not is_drift:
            for a in range(self.n):
                self.lambda_short[a] = 0.995 * self.lambda_short[a] + 0.005 * self.lambda_short_base
                self.lambda_long[a] = 0.995 * self.lambda_long[a] + 0.005 * self.lambda_long_base
        
        # Update short-window model with per-arm lambda
        lam_s = self.lambda_short[arm]
        self.A_short[arm] = lam_s * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = lam_s * self.b_short[arm] + reward * context
        
        # Update long-window model with per-arm lambda
        lam_l = self.lambda_long[arm]
        self.A_long[arm] = lam_l * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_l * self.b_long[arm] + reward * context
        
        # Periodic re-regularization to prevent numerical issues
        if self.t % 100 == 0:
            for a in range(self.n):
                self.A_short[a] = self.A_short[a] + self.reg * 0.1 * np.eye(self.d)
                self.A_long[a] = self.A_long[a] + self.reg * 0.1 * np.eye(self.d)
        
        # Update count
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
