"""Combined multi-timescale LinUCB with adaptive forgetting for non-stationary bandits."""
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
        
        # Regularization to ensure invertibility
        self.reg = 0.1
        
        # Short-window models (fast forgetting, high responsiveness)
        self.A_short = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        # Initial lambda for short window: aggressive forgetting
        self.lambda_short = [0.85 for _ in range(n_arms)]
        
        # Long-window models (slow forgetting, stability)
        self.A_long = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        # Initial lambda for long window: conservative forgetting
        self.lambda_long = [0.98 for _ in range(n_arms)]
        
        # Per-arm adaptive lambda parameters
        # Short window bounds
        self.lambda_short_min = 0.70
        self.lambda_short_max = 0.95
        self.lambda_short_base = 0.85
        # Long window bounds
        self.lambda_long_min = 0.90
        self.lambda_long_max = 0.995
        self.lambda_long_base = 0.98
        
        # Sensitivity of lambda adaptation to error
        self.error_sensitivity = 0.5
        
        # Residual tracking per arm per window
        # We track squared residuals to estimate variance/drift
        self.residuals_short = [deque(maxlen=20) for _ in range(n_arms)]
        self.residuals_long = [deque(maxlen=50) for _ in range(n_arms)]
        
        # Exploration parameter (UCB alpha)
        # We can adapt this too, but a fixed moderate value is often safer
        self.alpha = 1.2
        
        # Blend weight: how much to trust short vs long window
        # Higher weight on short means more responsive to changes
        self.blend_weight_short = 0.7
        self.blend_weight_long = 0.3
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _solve(self, A, b):
        """Solve A x = b efficiently with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_ucb_score(self, A, b, context, alpha):
        """Calculate UCB score for a given model and context."""
        # Solve for theta
        theta = self._solve(A, b)
        # Predicted mean
        mu = theta @ context
        
        # Compute variance term: context^T A^{-1} context
        try:
            A_inv_context = np.linalg.solve(A, context)
            v = context @ A_inv_context
        except np.linalg.LinAlgError:
            v = 1.0
            
        v = max(v, 0.0)
        ucb = mu + alpha * np.sqrt(v)
        return ucb, mu

    def _adapt_lambda(self, arm, residual_sq, is_short):
        """
        Adapt the forgetting factor lambda based on recent prediction errors.
        Higher recent errors imply non-stationarity, so we decrease lambda (forget faster).
        """
        if is_short:
            res_deque = self.residuals_short[arm]
            base = self.lambda_short_base
            l_min = self.lambda_short_min
            l_max = self.lambda_short_max
        else:
            res_deque = self.residuals_long[arm]
            base = self.lambda_long_base
            l_min = self.lambda_long_min
            l_max = self.lambda_long_max
            
        res_deque.append(residual_sq)
        
        if len(res_deque) < 5:
            # Not enough data to adapt reliably, keep base
            if is_short:
                self.lambda_short[arm] = base
            else:
                self.lambda_long[arm] = base
            return

        recent = list(res_deque)
        mean_res = np.mean(recent)
        
        # Map mean residual squared to lambda adjustment
        # If mean_res is small, lambda should be close to l_max (slow forgetting)
        # If mean_res is large, lambda should be close to l_min (fast forgetting)
        
        # Use a simple scaling factor
        # lambda = base / (1 + sensitivity * mean_res)
        # But we need to clamp to [l_min, l_max]
        
        # Let's use a linear mapping or a sigmoid-like approach
        # For simplicity, let's use:
        # adjustment = sensitivity * mean_res
        # new_lambda = base - adjustment
        # new_lambda = max(l_min, min(l_max, new_lambda))
        
        # However, mean_res can be very large. Let's normalize.
        # Assume typical reward scale is 1. If mean_res > 1, we are in high drift.
        # Let's use a soft clip.
        norm_res = mean_res / (1.0 + mean_res) # Maps [0, inf) to [0, 1)
        
        # Interpolate between l_min and l_max based on norm_res
        # If norm_res is 0 (no error), we want l_max (slow forgetting)
        # If norm_res is 1 (high error), we want l_min (fast forgetting)
        # So: lambda = l_max - (l_max - l_min) * norm_res
        
        new_lambda = l_max - (l_max - l_min) * norm_res
        
        if is_short:
            self.lambda_short[arm] = new_lambda
        else:
            self.lambda_long[arm] = new_lambda

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: return the arm with the highest count or just 0
            # To be safe and fast, return 0
            return 0
            
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        scores = []
        for a in range(self.n):
            # Get UCB from short model
            ucb_short, mu_short = self._get_ucb_score(self.A_short[a], self.b_short[a], context, self.alpha)
            
            # Get UCB from long model
            ucb_long, mu_long = self._get_ucb_score(self.A_long[a], self.b_long[a], context, self.alpha)
            
            # Blend the scores
            # We blend the UCB scores directly
            blended_score = self.blend_weight_short * ucb_short + self.blend_weight_long * ucb_long
            
            scores.append(blended_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.counts[arm] += 1
        
        # 1. Compute prediction for the pulled arm to get residuals
        # We use the pre-update models to calculate the error
        
        # Short window residual
        theta_short = self._solve(self.A_short[arm], self.b_short[arm])
        mu_pred_short = theta_short @ context
        residual_short = reward - mu_pred_short
        residual_sq_short = residual_short ** 2
        
        # Long window residual
        theta_long = self._solve(self.A_long[arm], self.b_long[arm])
        mu_pred_long = theta_long @ context
        residual_long = reward - mu_pred_long
        residual_sq_long = residual_long ** 2
        
        # 2. Adapt lambdas based on residuals
        self._adapt_lambda(arm, residual_sq_short, is_short=True)
        self._adapt_lambda(arm, residual_sq_long, is_short=False)
        
        # 3. Update A and b for the pulled arm using the adapted lambdas
        lam_short = self.lambda_short[arm]
        lam_long = self.lambda_long[arm]
        
        # Short window update
        self.A_short[arm] = lam_short * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = lam_short * self.b_short[arm] + reward * context
        
        # Long window update
        self.A_long[arm] = lam_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_long * self.b_long[arm] + reward * context
        
        # Note: We do NOT explicitly update the inverse or theta here to save time.
        # They are recomputed in select() using np.linalg.solve which is efficient.
        
# EVOLVE-BLOCK-END
