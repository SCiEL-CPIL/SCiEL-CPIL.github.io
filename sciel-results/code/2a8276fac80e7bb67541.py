"""
Non-stationary Linear Contextual Bandit Policy.

Uses a multi-timescale Linear UCB approach with RBF kernel augmentation:
1. RBF kernel features capture time-varying weight trajectories.
2. Short-window model (fast adaptation) for primary arm selection.
3. Long-window model (stable baseline) for regularization/backup.
4. Adaptive forgetting factors for both windows.
5. Maintains inverse matrices for efficient uncertainty computation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # RBF kernel parameters for time-dependent feature mapping
        self.n_rbf = 8  # Number of RBF basis functions
        self.rbf_centers = np.linspace(0, 10, self.n_rbf)  # Centers in normalized time
        self.rbf_widths = np.linspace(0.5, 3.0, self.n_rbf)  # Widths
        self.rbf_scale = 50.0  # Time scale for normalization
        
        # Augmented dimension: original dim + RBF features
        self.aug_dim = dim + self.n_rbf
        
        # Short-window models (fast) - track A, b, and A_inv in augmented space
        self.A_short = [np.eye(self.aug_dim) * 1.0 for _ in range(n_arms)]
        self.b_short = [np.zeros(self.aug_dim) for _ in range(n_arms)]
        self.A_inv_short = [np.eye(self.aug_dim) for _ in range(n_arms)]
        
        # Long-window models (slow)
        self.A_long = [np.eye(self.aug_dim) * 1.0 for _ in range(n_arms)]
        self.b_long = [np.zeros(self.aug_dim) for _ in range(n_arms)]
        self.A_inv_long = [np.eye(self.aug_dim) for _ in range(n_arms)]
        
        # Forgetting factors
        self.lambda_short_base = 0.93
        self.lambda_short_min = 0.60
        self.lambda_short_max = 0.99
        self.lambda_short = self.lambda_short_base
        
        self.lambda_long_base = 0.995
        self.lambda_long_min = 0.95
        self.lambda_long_max = 0.9999
        self.lambda_long = self.lambda_long_base
        
        # Error tracking
        self.error_buffer_short = np.zeros(30)
        self.error_idx_short = 0
        self.error_count_short = 0
        self.expected_var_short = 1.0
        
        self.error_buffer_long = np.zeros(50)
        self.error_idx_long = 0
        self.error_count_long = 0
        self.expected_var_long = 1.0
        
        # Exploration
        self.alpha_start = 1.2
        self.alpha_min = 0.3
        self.decay_rate = 0.003
        
        # Time
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Reset thresholds
        self.reset_threshold = 2.5
        
        # Model disagreement tracking
        self.disagreement_buffer = np.zeros(20)
        self.disagreement_idx = 0
        self.disagreement_count = 0
        self.avg_disagreement = 0.0

    def _update_lambda_short(self, sq_error):
        """Update short-window forgetting factor."""
        self.error_buffer_short[self.error_idx_short] = sq_error
        self.error_idx_short = (self.error_idx_short + 1) % len(self.error_buffer_short)
        if self.error_count_short < len(self.error_buffer_short):
            self.error_count_short += 1
            
        if self.error_count_short == 0:
            return
            
        avg_sq_err = np.mean(self.error_buffer_short[:self.error_count_short])
        error_ratio = avg_sq_err / (self.expected_var_short + 1e-8)
        penalty = error_ratio / (1.0 + error_ratio)
        new_lambda = self.lambda_short_max - (self.lambda_short_max - self.lambda_short_min) * penalty
        self.lambda_short = 0.9 * self.lambda_short + 0.1 * new_lambda
        self.expected_var_short = 0.99 * self.expected_var_short + 0.01 * (avg_sq_err + 0.1)
        
    def _compute_rbf_features(self, t):
        """Compute RBF kernel features for time t."""
        t_norm = t / self.rbf_scale
        features = np.zeros(self.n_rbf)
        for i in range(self.n_rbf):
            features[i] = np.exp(-0.5 * ((t_norm - self.rbf_centers[i]) / self.rbf_widths[i]) ** 2)
        return features
    
    def _augment_context(self, context, t):
        """Augment context with RBF time features."""
        ctx = np.asarray(context, dtype=np.float64)
        rbf = self._compute_rbf_features(t)
        return np.concatenate([ctx, rbf])

    def _update_lambda_long(self, sq_error):
        """Update long-window forgetting factor."""
        self.error_buffer_long[self.error_idx_long] = sq_error
        self.error_idx_long = (self.error_idx_long + 1) % len(self.error_buffer_long)
        if self.error_count_long < len(self.error_buffer_long):
            self.error_count_long += 1
            
        if self.error_count_long == 0:
            return
            
        avg_sq_err = np.mean(self.error_buffer_long[:self.error_count_long])
        error_ratio = avg_sq_err / (self.expected_var_long + 1e-8)
        penalty = error_ratio / (1.0 + error_ratio)
        new_lambda = self.lambda_long_max - (self.lambda_long_max - self.lambda_long_min) * penalty
        self.lambda_long = 0.95 * self.lambda_long + 0.05 * new_lambda
        self.expected_var_long = 0.999 * self.expected_var_long + 0.001 * (avg_sq_err + 0.1)
        
    def _update_disagreement(self, mu_s, mu_l):
        """Track model disagreement for adaptive exploration."""
        self.disagreement_buffer[self.disagreement_idx] = abs(mu_s - mu_l)
        self.disagreement_idx = (self.disagreement_idx + 1) % len(self.disagreement_buffer)
        if self.disagreement_count < len(self.disagreement_buffer):
            self.disagreement_count += 1
        
        if self.disagreement_count > 0:
            self.avg_disagreement = np.mean(self.disagreement_buffer[:self.disagreement_count])

    def _reset_arm_short(self, arm, context, reward):
        """Reset short-window model for an arm."""
        self.A_short[arm] = np.eye(self.aug_dim) * 1.0
        self.b_short[arm] = reward * context
        self.A_inv_short[arm] = np.eye(self.aug_dim)

    def _reset_arm_long(self, arm, context, reward):
        """Reset long-window model for an arm."""
        self.A_long[arm] = np.eye(self.aug_dim) * 1.0
        self.b_long[arm] = reward * context
        self.A_inv_long[arm] = np.eye(self.aug_dim)

    def select(self, context):
        # Deadline check - use greedy on short model
        if time.time() > self.deadline:
             best_arm = 0
             best_score = -np.inf
             ctx = self._augment_context(context, self.t)
             for a in range(self.n_arms):
                 theta = self.A_inv_short[a] @ self.b_short[a]
                 mu = theta @ ctx
                 if mu > best_score:
                     best_score = mu
                     best_arm = a
             return best_arm

        self.t += 1
        ctx = self._augment_context(context, self.t)
        
        # Adaptive alpha based on disagreement
        base_alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        # Boost exploration when models disagree
        disagreement_boost = 1.0 + 0.5 * min(self.avg_disagreement, 2.0)
        alpha = base_alpha * disagreement_boost
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Short-window model
            A_inv_s = self.A_inv_short[a]
            b_s = self.b_short[a]
            theta_s = A_inv_s @ b_s
            mu_s = theta_s @ ctx
            
            # Uncertainty from short model: v = ctx^T A_inv ctx
            v_s = ctx @ (A_inv_s @ ctx)
            v_s = max(v_s, 0.0)
            
            # Long-window model
            A_inv_l = self.A_inv_long[a]
            b_l = self.b_long[a]
            theta_l = A_inv_l @ b_l
            mu_l = theta_l @ ctx
            
            v_l = ctx @ (A_inv_l @ ctx)
            v_l = max(v_l, 0.0)
            
            # Track disagreement
            self._update_disagreement(mu_s, mu_l)
            
            # Inverse-variance blending: weight by 1/(1+v)
            w_s = 1.0 / (1.0 + v_s)
            w_l = 1.0 / (1.0 + v_l)
            w_sum = w_s + w_l
            w_s_norm = w_s / w_sum
            w_l_norm = w_l / w_sum
            
            mu_blend = w_s_norm * mu_s + w_l_norm * mu_l
            
            # Combined uncertainty: weighted combination
            v_blend = w_s_norm * v_s + w_l_norm * v_l
            
            score = mu_blend + alpha * np.sqrt(v_blend)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        ctx = self._augment_context(context, self.t)
        
        # --- Update short-window model ---
        A_inv_s = self.A_inv_short[arm]
        b_s = self.b_short[arm]
        
        # Compute prediction and uncertainty before update
        theta_s = A_inv_s @ b_s
        mu_pred_s = theta_s @ ctx
        v_s = ctx @ (A_inv_s @ ctx)
        v_s = max(v_s, 0.0)
        std_pred_s = np.sqrt(v_s) + np.sqrt(self.expected_var_short)
        
        residual_s = reward - mu_pred_s
        sq_error_s = residual_s ** 2
        
        self._update_lambda_short(sq_error_s)
        lam_s = self.lambda_short
        
        # Check for shock reset on short model
        if abs(residual_s) > self.reset_threshold * std_pred_s:
            self._reset_arm_short(arm, ctx, reward)
        else:
            # Update A and b with forgetting
            outer = np.outer(ctx, ctx)
            self.A_short[arm] = lam_s * self.A_short[arm] + outer
            self.b_short[arm] = lam_s * b_s + reward * ctx
            
            # Update A_inv using Sherman-Morrison-like approach
            A_inv_old = A_inv_s
            x = ctx
            lam = lam_s
            
            # Compute x^T A_inv x
            Ax = A_inv_old @ x
            xAx = x @ Ax  # scalar
            
            denom = 1.0 + xAx / lam
            if abs(denom) < 1e-12:
                # Fallback: recompute inverse
                try:
                    self.A_inv_short[arm] = np.linalg.inv(self.A_short[arm])
                except:
                    self.A_inv_short[arm] = np.eye(self.aug_dim)
            else:
                term1 = (1.0 / lam) * A_inv_old
                term2 = (1.0 / (lam * lam)) * (np.outer(Ax, Ax) / denom)
                self.A_inv_short[arm] = term1 - term2
        
        # --- Update long-window model ---
        A_inv_l = self.A_inv_long[arm]
        b_l = self.b_long[arm]
        
        theta_l = A_inv_l @ b_l
        mu_pred_l = theta_l @ ctx
        v_l = ctx @ (A_inv_l @ ctx)
        v_l = max(v_l, 0.0)
        std_pred_l = np.sqrt(v_l) + np.sqrt(self.expected_var_long)
        
        residual_l = reward - mu_pred_l
        sq_error_l = residual_l ** 2
        
        self._update_lambda_long(sq_error_l)
        lam_l = self.lambda_long
        
        # Less aggressive reset for long model
        if abs(residual_l) > 4.0 * std_pred_l:
            self._reset_arm_long(arm, ctx, reward)
        else:
            outer = np.outer(ctx, ctx)
            self.A_long[arm] = lam_l * self.A_long[arm] + outer
            self.b_long[arm] = lam_l * b_l + reward * ctx
            
            A_inv_old = A_inv_l
            x = ctx
            lam = lam_l
            
            Ax = A_inv_old @ x
            xAx = x @ Ax
            
            denom = 1.0 + xAx / lam
            if abs(denom) < 1e-12:
                try:
                    self.A_inv_long[arm] = np.linalg.inv(self.A_long[arm])
                except:
                    self.A_inv_long[arm] = np.eye(self.aug_dim)
            else:
                term1 = (1.0 / lam) * A_inv_old
                term2 = (1.0 / (lam * lam)) * (np.outer(Ax, Ax) / denom)
                self.A_inv_long[arm] = term1 - term2

# EVOLVE-BLOCK-END
