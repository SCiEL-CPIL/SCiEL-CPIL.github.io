import numpy as np
import time
from collections import deque

class Policy:
    """
    Multi-timescale Linear Thompson Sampling with per-arm adaptive forgetting.
    
    Combines the Bayesian Thompson Sampling structure (from current program) with
    the multi-timescale architecture and adaptive lambda machinery (from inspirations).
    Maintains short and long window posterior estimates per arm with independently
    adapted discount factors. Uses Thompson Sampling for exploration/exploitation.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        self.reg = 0.1
        
        # Short-window models (fast forgetting for adaptation)
        self.A_short = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_short = [0.90 for _ in range(n_arms)]
        
        # Long-window models (slow forgetting for stability)
        self.A_long = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lambda_long = [0.98 for _ in range(n_arms)]
        
        # Per-arm adaptive lambda parameters
        self.lambda_short_base = 0.90
        self.lambda_short_min = 0.75
        self.lambda_short_max = 0.95
        self.lambda_long_base = 0.98
        self.lambda_long_min = 0.92
        self.lambda_long_max = 0.995
        
        # Residual tracking per arm per window
        self.residuals_short = [deque(maxlen=20) for _ in range(n_arms)]
        self.residuals_long = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Cholesky factors cached
        self._chol_short = [None] * n_arms
        self._chol_long = [None] * n_arms
        
        for a in range(n_arms):
            self._chol_short[a] = np.linalg.cholesky(self.A_short[a])
            self._chol_long[a] = np.linalg.cholesky(self.A_long[a])
        
        # Blend weight for short vs long window in Thompson Sampling
        self.blend_weight = 0.6
        self.blend_weight_min = 0.4
        self.blend_weight_max = 0.85
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0

    def _solve(self, A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_lambda(self, arm, residual_sq, is_short):
        """Update per-arm lambda based on recent residuals using robust scaling."""
        if is_short:
            self.residuals_short[arm].append(residual_sq)
            res = self.residuals_short[arm]
        else:
            self.residuals_long[arm].append(residual_sq)
            res = self.residuals_long[arm]
        
        if len(res) < 5:
            return
        
        recent = np.array(list(res))
        mean_res = np.mean(recent)
        std_res = np.std(recent)
        
        # MAD-based robust scaling
        median_res = np.median(recent)
        mad = np.median(np.abs(recent - median_res))
        scale = max(mad * 1.4826, 0.05)
        
        error_metric = (mean_res + 0.5 * std_res) / scale
        
        if is_short:
            target = self.lambda_short_base - 0.03 * np.log1p(error_metric)
            target = max(self.lambda_short_min, min(self.lambda_short_max, target))
            self.lambda_short[arm] = 0.9 * self.lambda_short[arm] + 0.1 * target
        else:
            target = self.lambda_long_base - 0.015 * np.log1p(error_metric)
            target = max(self.lambda_long_min, min(self.lambda_long_max, target))
            self.lambda_long[arm] = 0.95 * self.lambda_long[arm] + 0.05 * target

    def _recompute_cholesky(self, arm, is_short):
        """Recompute Cholesky factor for the specified arm and window."""
        if is_short:
            A = self.A_short[arm]
        else:
            A = self.A_long[arm]
        
        # Ensure positive definite
        A = A + 1e-8 * np.eye(self.d)
        if is_short:
            self.A_short[arm] = A
        else:
            self.A_long[arm] = A
        
        try:
            chol = np.linalg.cholesky(A)
            if is_short:
                self._chol_short[arm] = chol
            else:
                self._chol_long[arm] = chol
        except np.linalg.LinAlgError:
            # Fallback: reset to regularized identity
            A_reset = np.eye(self.d) * self.reg
            if is_short:
                self.A_short[arm] = A_reset
                self._chol_short[arm] = np.linalg.cholesky(A_reset)
            else:
                self.A_long[arm] = A_reset
                self._chol_long[arm] = np.linalg.cholesky(A_reset)

    def _sample_theta(self, arm, is_short):
        """
        Sample theta from posterior N(theta_hat, A^{-1}) using Cholesky.
        Returns the sampled theta vector.
        """
        if is_short:
            A = self.A_short[arm]
            b = self.b_short[arm]
            L = self._chol_short[arm]
        else:
            A = self.A_long[arm]
            b = self.b_long[arm]
            L = self._chol_long[arm]
        
        # Compute mean theta = A^{-1} b
        try:
            # Solve L y = b, then theta = L^{-T} y
            y = np.linalg.solve(L, b)
            theta = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
        
        # Sample from posterior: theta_sample = theta + L^{-1} z
        z = np.random.randn(self.d)
        try:
            v = np.linalg.solve(L, z)
            theta_sample = theta + v
        except np.linalg.LinAlgError:
            theta_sample = theta
        
        return theta_sample

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return arm with highest mean estimate from long-window
            context = np.asarray(context, dtype=np.float64)
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                try:
                    L = self._chol_long[a]
                    y = np.linalg.solve(L, self.b_long[a])
                    theta = np.linalg.solve(L.T, y)
                    mean = theta @ context
                    if mean > best_mean:
                        best_mean = mean
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return int(best_arm)
        
        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling with blended short/long window
        scores = []
        for a in range(self.n):
            # Sample from both short and long window posteriors
            theta_short = self._sample_theta(a, is_short=True)
            theta_long = self._sample_theta(a, is_short=False)
            
            # Blend the samples
            w = self.blend_weight
            theta_blend = w * theta_short + (1.0 - w) * theta_long
            
            score = theta_blend @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        self.t += 1
        self.counts[arm] += 1
        
        # Predicted rewards before update
        pred_short = None
        pred_long = None
        try:
            L_s = self._chol_short[arm]
            y_s = np.linalg.solve(L_s, self.b_short[arm])
            theta_s = np.linalg.solve(L_s.T, y_s)
            pred_short = theta_s @ context
        except np.linalg.LinAlgError:
            pred_short = 0.0
        
        try:
            L_l = self._chol_long[arm]
            y_l = np.linalg.solve(L_l, self.b_long[arm])
            theta_l = np.linalg.solve(L_l.T, y_l)
            pred_long = theta_l @ context
        except np.linalg.LinAlgError:
            pred_long = 0.0
        
        # Update short-window
        gamma_s = self.lambda_short[arm]
        self.A_short[arm] = gamma_s * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = gamma_s * self.b_short[arm] + reward * context
        self._recompute_cholesky(arm, is_short=True)
        
        # Update long-window
        gamma_l = self.lambda_long[arm]
        self.A_long[arm] = gamma_l * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = gamma_l * self.b_long[arm] + reward * context
        self._recompute_cholesky(arm, is_short=False)
        
        # Compute residuals for adaptive lambda
        residual_short_sq = (reward - pred_short) ** 2
        residual_long_sq = (reward - pred_long) ** 2
        
        self._update_lambda(arm, residual_short_sq, is_short=True)
        self._update_lambda(arm, residual_long_sq, is_short=False)
        
        # Adaptively adjust blend weight based on recent performance
        if self.t > 50:
            # Use longer-term residual comparison to adjust blend
            if len(self.residuals_short[arm]) >= 5 and len(self.residuals_long[arm]) >= 5:
                mean_res_s = np.mean(list(self.residuals_short[arm]))
                mean_res_l = np.mean(list(self.residuals_long[arm]))
                # If short window has lower residuals, increase its weight
                if mean_res_s < mean_res_l:
                    self.blend_weight = min(self.blend_weight_max, self.blend_weight + 0.01)
                elif mean_res_l < mean_res_s:
                    self.blend_weight = max(self.blend_weight_min, self.blend_weight - 0.01)
