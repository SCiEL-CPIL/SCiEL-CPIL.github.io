import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Strategy:
    - Per-arm ridge regression with an adaptive forgetting factor.
    - SMW rank-1 inverse updates for speed and stability.
    - Forgetting factor tuned per-arm based on recent prediction errors.
    - Exploration via UCB with a conservative adaptive alpha.
    - Deadline guard to ensure we finish within the time budget.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm state: A (dim x dim), b (dim,)
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) / 0.1 for _ in range(n_arms)]
        
        # Forgetting: per-arm lambda in [lam_min, lam_max]
        self.lam_min = 0.93
        self.lam_max = 0.97
        self.error_windows = [deque(maxlen=15) for _ in range(n_arms)]
        
        # Exploration: conservative alpha
        self.alpha_min = 0.3
        self.alpha_max = 1.0
        self.alpha = 0.5
        
        # Global error tracking
        self.global_errors = deque(maxlen=30)
        
        # Timing
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin
        
        # Step counter
        self.t = 0
        
        # MSE scale for lambda mapping
        self.mse_scale = 0.05
        
        # Periodic re-inverse interval to prevent drift
        self.reinv_interval = 50

    def _elapsed(self):
        return time.time() - self.start_time

    def _select_lambda(self, arm):
        """Choose forgetting factor based on recent errors for this arm."""
        errs = self.error_windows[arm]
        if len(errs) == 0:
            return self.lam_max
        mse = np.mean(np.array(list(errs)) ** 2)
        # Map MSE to lambda: high MSE -> low lambda (fast forgetting)
        norm = mse / (mse + self.mse_scale)
        lam = self.lam_max - norm * (self.lam_max - self.lam_min)
        return max(self.lam_min, min(self.lam_max, lam))

    def _update_alpha(self):
        """Adapt alpha based on global error level."""
        if len(self.global_errors) < 5:
            return
        mse = np.mean(np.array(list(self.global_errors)) ** 2)
        # Target alpha: more exploration when errors are high
        target = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.1))
        # Smooth update
        self.alpha = 0.85 * self.alpha + 0.15 * target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _recompute_inverse(self, arm):
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.inv_A[arm] = np.eye(self.d) / 0.1

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        if self._elapsed() > self.deadline:
            # Fallback: greedy based on current estimates
            scores = []
            for a in range(self.n):
                theta = self.inv_A[a] @ self.b[a]
                scores.append(float(theta @ context))
            return int(np.argmax(scores))
        
        scores = []
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            mu = float(theta @ context)
            var = float(context @ self.inv_A[a] @ context)
            bonus = self.alpha * np.sqrt(max(var, 0.0))
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        if self._elapsed() > self.deadline:
            return
        
        self.t += 1
        
        # Compute residual using current model
        theta = self.inv_A[arm] @ self.b[arm]
        pred = float(theta @ context)
        residual = reward - pred
        
        # Update error windows
        self.error_windows[arm].append(residual)
        self.global_errors.append(residual)
        
        # Determine forgetting factor
        lam = self._select_lambda(arm)
        
        # Update A and b with forgetting
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # SMW rank-1 inverse update
        # inv(lam*A_old + xx^T) where inv_B = (1/lam)*inv_A_old
        inv_B = (1.0 / lam) * self.inv_A[arm]
        inv_B_x = inv_B @ context
        denom = 1.0 + float(context @ inv_B_x)
        
        if abs(denom) < 1e-12:
            # Numerical issue, fall back to direct inverse
            self._recompute_inverse(arm)
        else:
            self.inv_A[arm] = inv_B - (np.outer(inv_B_x, inv_B_x) / denom)
        
        # Periodic full re-inverse to prevent drift
        if self.t % self.reinv_interval == 0:
            self._recompute_inverse(arm)
        
        # Update exploration parameter
        self._update_alpha()
