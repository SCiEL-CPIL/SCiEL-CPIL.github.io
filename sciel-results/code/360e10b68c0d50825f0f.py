"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """Multi-timescale LinUCB: short-window for adaptation, long-window for stability."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Short-window model (fast adaptation)
        reg_short = 0.5
        self.A_inv_short = [np.eye(dim) / reg_short for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_short = 0.95  # Faster decay for short window
        
        # Long-window model (stable baseline)
        reg_long = 0.5
        self.A_inv_long = [np.eye(dim) / reg_long for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_long = 0.99  # Slower decay for long window
        
        # History for error tracking
        self.error_hist = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha_base = 1.2
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin

    def _select_arm(self, context):
        """Select arm using combined short/long window UCB."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        for a in range(self.n):
            # Short window prediction
            A_inv_s = self.A_inv_short[a]
            b_s = self.b_short[a]
            theta_s = A_inv_s @ b_s
            mu_s = theta_s @ context
            v_s = context @ A_inv_s @ context
            
            # Long window prediction
            A_inv_l = self.A_inv_long[a]
            b_l = self.b_long[a]
            theta_l = A_inv_l @ b_l
            mu_l = theta_l @ context
            v_l = context @ A_inv_l @ context
            
            # Combined estimate: blend short and long
            # Use short window primarily for adaptation, long for regularization
            mu_combined = 0.7 * mu_s + 0.3 * mu_l
            v_combined = 0.7 * v_s + 0.3 * v_l
            
            # Adaptive alpha based on recent error variance
            if len(self.error_hist[a]) > 0:
                avg_err_sq = np.mean(self.error_hist[a])
                # Increase exploration when errors are high (regime change)
                alpha_adaptive = self.alpha_base * (1.0 + 0.5 * avg_err_sq)
            else:
                alpha_adaptive = self.alpha_base
            
            # UCB score
            score = mu_combined + alpha_adaptive * np.sqrt(max(v_combined, 0.0))
            scores.append(score)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0

        arm = self._select_arm(context)
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Compute prediction residual using short window
        A_inv_s = self.A_inv_short[arm]
        b_s = self.b_short[arm]
        theta_s = A_inv_s @ b_s
        mu_pred = theta_s @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # Update error history
        self.error_hist[arm].append(residual_sq)

        # Update short window (fast decay)
        lam_s = self.lambda_short
        A_inv_s_old = self.A_inv_short[arm]
        A_inv_x_s = A_inv_s_old @ context
        denom_s = lam_s + context @ A_inv_x_s
        correction_s = np.outer(A_inv_x_s, A_inv_x_s) / denom_s
        self.A_inv_short[arm] = (1.0 / lam_s) * (A_inv_s_old - correction_s)
        self.b_short[arm] = lam_s * self.b_short[arm] + reward * context

        # Update long window (slow decay)
        lam_l = self.lambda_long
        A_inv_l_old = self.A_inv_long[arm]
        A_inv_x_l = A_inv_l_old @ context
        denom_l = lam_l + context @ A_inv_x_l
        correction_l = np.outer(A_inv_x_l, A_inv_x_l) / denom_l
        self.A_inv_long[arm] = (1.0 / lam_l) * (A_inv_l_old - correction_l)
        self.b_long[arm] = lam_l * self.b_long[arm] + reward * context

# EVOLVE-BLOCK-END
