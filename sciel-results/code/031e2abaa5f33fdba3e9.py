import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear contextual bandits.
    Uses Sherman-Morrison-Woodbury for efficient rank-1 updates.
    Adaptive per-arm forgetting factor based on recent prediction errors.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as identity (for RLS-like stability)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached inverses
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.90
        self.lambda_max = 0.995
        self.error_windows = [deque(maxlen=30) for _ in range(n_arms)]
        
        # Exploration
        self.alpha_base = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.5
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 150s limit (evaluator kills at 150s)
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=100)

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _smw_update(self, arm, context):
        """
        Update A and b with rank-1 update, then update inv_A using Sherman-Morrison.
        A_new = lam * A_old + x x^T
        b_new = lam * b_old + r * x
        
        We need inv(A_new).
        Let A_old_inv = inv(A_old).
        A_new = lam * A_old + x x^T = lam * (A_old + (1/lam) x x^T)
        Let M = A_old + (1/lam) x x^T
        Then A_new = lam * M => inv(A_new) = (1/lam) * inv(M)
        
        inv(M) = inv(A_old + (1/lam) x x^T)
        Using SMW: inv(A + u v^T) = inv(A) - inv(A) u v^T inv(A) / (1 + v^T inv(A) u)
        Here u = x / sqrt(lam), v = x / sqrt(lam)
        So inv(M) = inv_A - (inv_A @ (x/sqrt(lam))) @ (x/sqrt(lam))^T @ inv_A / (1 + (x/sqrt(lam))^T inv_A (x/sqrt(lam)))
        
        Let c = 1/sqrt(lam)
        u = c * x
        v = c * x
        denom = 1 + v^T inv_A u = 1 + c^2 * (x^T inv_A x)
        numerator = c^2 * (inv_A x) (x^T inv_A)
        
        inv(M) = inv_A - numerator / denom
        inv(A_new) = (1/lam) * inv(M)
        """
        lam = self._get_lambda(arm)
        c = 1.0 / np.sqrt(lam)
        u = c * context
        v = c * context
        
        inv_A = self.inv_A[arm]
        b = self.b[arm]
        A = self.A[arm]
        
        # Compute x^T inv_A x
        Ax = inv_A @ context
        quad = context @ Ax  # scalar
        
        # denom
        denom = 1.0 + (lam * quad)  # 1 + c^2 * quad = 1 + (1/lam)*quad? No.
        # Wait: v^T inv_A u = (c x)^T inv_A (c x) = c^2 x^T inv_A x = (1/lam) * quad
        # So denom = 1 + quad/lam
        
        # Let's re-derive carefully.
        # A_new = lam A + x x^T
        # We want inv(A_new).
        # A_new = lam (A + (1/lam) x x^T)
        # Let B = A + (1/lam) x x^T
        # inv(B) = inv(A) - inv(A) (1/lam x) (1/lam x)^T inv(A) / (1 + (1/lam x)^T inv(A) (1/lam x))
        # = inv_A - (1/lam^2) (inv_A x) (x^T inv_A) / (1 + (1/lam) x^T inv_A x)
        # = inv_A - (inv_A x) (x^T inv_A) / (lam * (lam + x^T inv_A x))  <-- This is getting messy.
        
        # Simpler: Just compute inv(A_new) directly if dim is small?
        # Dim is likely small (5 arms, dim probably 5-10).
        # np.linalg.inv is O(d^3). If d=10, it's very fast.
        # SMW is O(d^2).
        # Let's stick to SMW for safety but implement correctly.
        
        # A_new = lam * A + outer(x, x)
        self.A[arm] = lam * A + np.outer(context, context)
        self.b[arm] = lam * b + reward * context  # reward is passed in update, but here we only update A
        
        # Actually, update() calls this. Let's pass reward to update inv_A? No, inv_A only depends on A.
        # But we need to update b as well.
        
        # Let's just recompute inv if dim is small enough, or use SMW.
        # For robustness, let's use np.linalg.inv if dim <= 10, else SMW.
        # But SMW is better.
        
        # Correct SMW for A_new = lam * A_old + x x^T
        # inv(A_new) = (1/lam) * inv(A_old + (1/lam) x x^T)
        # Let u = x, v = (1/lam) x
        # inv(A + u v^T) = inv(A) - (inv(A) u v^T inv(A)) / (1 + v^T inv(A) u)
        # Here A is A_old, u is x, v is (1/lam) x
        # inv(A_old + (1/lam) x x^T) = inv_A - (inv_A x) ((1/lam) x^T inv_A) / (1 + (1/lam) x^T inv_A x)
        # = inv_A - (1/lam) (inv_A x) (x^T inv_A) / (1 + (1/lam) quad)
        # = inv_A - (inv_A x) (x^T inv_A) / (lam + quad)
        
        # So inv(A_new) = (1/lam) * [ inv_A - (inv_A x) (x^T inv_A) / (lam + quad) ]
        # = (1/lam) inv_A - (1/lam) (inv_A x) (x^T inv_A) / (lam + quad)
        
        # Let's code this.
        quad = context @ (inv_A @ context)
        denom = lam + quad
        if denom < 1e-10:
            # Fallback
            self._recompute_inv(arm)
            return
        
        term = np.outer(inv_A @ context, context @ inv_A)  # (inv_A x) (x^T inv_A)
        new_inv = (1.0 / lam) * inv_A - (1.0 / lam) * term / denom
        
        self.inv_A[arm] = new_inv

    def _recompute_inv(self, arm):
        """Recompute inverse of A[arm] and cache it."""
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.inv_A[arm] = np.eye(self.d)

    def _get_lambda(self, arm):
        """Determine forgetting factor based on recent errors for this arm."""
        if len(self.error_windows[arm]) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows[arm])
        mse = np.mean(errors**2)
        
        # Map MSE to lambda
        scale = 0.1
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance."""
        if len(self.global_errors) < 10:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.2))
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick best mean
            scores = []
            for a in range(self.n):
                theta = self.inv_A[a] @ self.b[a]
                mu = theta @ context
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            mu = theta @ context
            var = context @ self.inv_A[a] @ context
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Calculate residual
        theta_old = self.inv_A[arm] @ self.b[arm]
        pred = theta_old @ context
        residual = reward - pred
        
        # 2. Store residual
        self.error_windows[arm].append(residual)
        self.global_errors.append(residual)
        
        # 3. Determine lambda
        lam = self._get_lambda(arm)
        
        # 4. Update A and b
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # 5. Update inv_A using SMW
        # We need to pass context and lam to smw update logic
        inv_A = self.inv_A[arm]
        Ax = inv_A @ context
        quad = context @ Ax
        denom = lam + quad
        if denom > 1e-10:
            term = np.outer(Ax, context @ inv_A)
            self.inv_A[arm] = (1.0 / lam) * inv_A - (1.0 / lam) * term / denom
        else:
            self._recompute_inv(arm)
        
        # 6. Update alpha
        self._update_alpha()
