import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Strategy:
    - Per-arm ridge regression with an adaptive forgetting factor.
    - The forgetting factor is tuned per-arm based on recent prediction errors:
      high error -> faster forgetting (lower lambda), low error -> slower forgetting (higher lambda).
    - Exploration via UCB with an adaptive alpha that responds to global error level.
    - Deadline guard to ensure we finish within the time budget.
    - Uses Sherman-Morrison or direct inverse updates for efficiency.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm state: A (dim x dim), b (dim,)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        
        # Forgetting: per-arm lambda in [lam_min, lam_max]
        self.lam_min = 0.90
        self.lam_max = 0.998
        self.error_windows = [deque(maxlen=15) for _ in range(n_arms)]
        self.error_win_size = 15
        
        # Exploration: adaptive alpha
        self.alpha_min = 0.4
        self.alpha_max = 2.0
        self.alpha = 1.0
        
        # Global error tracking
        self.global_errors = deque(maxlen=30)
        self.global_win_size = 30
        
        # Timing
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin
        
        # Step counter
        self.t = 0
        
        # Precompute small scale for lambda mapping
        self.mse_scale = 0.1

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
        target = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.2))
        # Smooth update
        self.alpha = 0.8 * self.alpha + 0.2 * target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _recompute_inverse(self, arm):
        """Use Sherman-Morrison-Woodbury for rank-1 update of inverse."""
        A = self.A[arm]
        inv_A = self.inv_A[arm]
        x = None  # We need the context vector for SMW
        # Actually, we need to pass context. Let's restructure.
        pass

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
        self.error_windows[arm] = np.append(self.error_windows[arm], residual)
        if len(self.error_windows[arm]) > self.error_win_size:
            self.error_windows[arm] = self.error_windows[arm][-self.error_win_size:]
        
        self.global_errors.append(residual)
        if len(self.global_errors) > self.global_win_size:
            self.global_errors = self.global_errors[-self.global_win_size:]
        
        # Determine forgetting factor
        lam = self._select_lambda(arm)
        
        # Update A and b with forgetting
        old_A = self.A[arm]
        self.A[arm] = lam * old_A + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # Sherman-Morrison-Woodbury update for inverse
        # inv(A + xx^T) = inv(A) - inv(A)xx^T inv(A) / (1 + x^T inv(A) x)
        # But we have lam*A + xx^T, not A + xx^T
        # inv(lam*A + xx^T) = inv(lam*A + xx^T)
        # = (1/lam) * inv(A + (1/lam) xx^T)
        # = (1/lam) * [inv(A) - inv(A)(1/lam)x x^T inv(A) / (1 + (1/lam) x^T inv(A) x)]
        # = (1/lam) * inv(A) - (1/lam^2) * inv(A) x x^T inv(A) / (1 + (1/lam) x^T inv(A) x)
        
        inv_old = self.inv_A[arm]
        x = context
        Ax = inv_old @ x
        scalar = 1.0 + (1.0 / lam) * float(x @ Ax)
        
        if abs(scalar) < 1e-10:
            # Fallback to direct inversion
            try:
                self.inv_A[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.inv_A[arm] = np.eye(self.d)
        else:
            self.inv_A[arm] = (1.0 / lam) * inv_old - (1.0 / (lam * lam)) * np.outer(Ax, Ax) / scalar
        
        # Update exploration parameter
        self._update_alpha()
