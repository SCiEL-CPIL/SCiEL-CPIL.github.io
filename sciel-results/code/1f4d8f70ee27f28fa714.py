"""Adaptive linear bandit policy for non-stationary settings."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Per-arm linear bandit with adaptive forgetting.
    
    Model: reward = x^T theta_a + noise
    - Each arm has its own linear parameter vector theta_a.
    - Adaptive forgetting factor reacts to prediction errors.
    - Adaptive UCB alpha based on recent error magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm linear models
        # A[a]: dim x dim matrix
        # b[a]: dim vector
        # inv_A[a]: dim x dim matrix (inverse of A[a])
        self.A = np.array([np.eye(dim) for _ in range(n_arms)])
        self.b = np.array([np.zeros(dim) for _ in range(n_arms)])
        self.inv_A = np.array([np.eye(dim) for _ in range(n_arms)])

        # Shared global linear model for sample efficiency
        self.A_g = np.eye(dim)
        self.b_g = np.zeros(dim)
        self.inv_A_g = np.eye(dim)

        # Adaptive forgetting
        self.base_lambda = 0.92
        self.min_lambda = 0.78
        self.max_lambda = 0.99
        self.error_buffer = deque(maxlen=60)

        # Adaptive UCB
        self.base_alpha = 1.4
        self.min_alpha = 0.5
        self.max_alpha = 3.5

        # Blend weight between global and per-arm predictions
        self.global_blend = 0.3

        # Track predictions for error computation
        self._last_mu = np.zeros(n_arms)

        # Time guard
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0

        # Round counter
        self._round = 0

        # Track per-arm sample counts for confidence weighting
        self.arm_counts = np.zeros(n_arms)

    def _compute_lambda(self):
        """Adaptive forgetting factor based on recent prediction errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Use a robust scale: median absolute deviation or similar
        # err_scale adapts to the problem's natural error magnitude
        # Use a running estimate of typical error scale
        if len(recent) >= 10:
            # Use interquartile range as robust scale
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + std_err * 0.5) / err_scale

        if normalized < 0.08:
            return self.max_lambda
        elif normalized > 2.0:
            return self.min_lambda
        else:
            # Smooth interpolation with slightly sharper transition
            t = (normalized - 0.08) / (2.0 - 0.08)
            # Use a smoothstep-like curve for smoother transitions
            t = t * t * (3.0 - 2.0 * t)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _compute_alpha(self):
        """Adaptive exploration parameter."""
        if len(self.error_buffer) < 10:
            return self.base_alpha

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Robust error scale
        if len(recent) >= 10:
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + 0.5 * std_err) / err_scale

        if normalized < 0.12:
            return self.min_alpha
        elif normalized > 1.8:
            return self.max_alpha
        else:
            t = (normalized - 0.12) / (1.8 - 0.12)
            # Smoothstep for smoother exploration transitions
            t = t * t * (3.0 - 2.0 * t)
            return self.min_alpha + t * (self.max_alpha - self.min_alpha)

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        # Get current adaptive parameters
        alpha = self._compute_alpha()

        # Compute global model prediction
        theta_g = self.inv_A_g @ self.b_g
        mu_global = theta_g @ context
        var_global = context @ (self.inv_A_g @ context)
        std_global = np.sqrt(max(0.0, var_global))

        # Adaptive blend: increase global influence as more data accumulates
        n_total = max(1, self._round)
        blend = self.global_blend * min(1.0, n_total / 50.0)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Compute theta_a
            theta_a = self.inv_A[a] @ self.b[a]
            
            # Per-arm prediction
            mu_arm = theta_a @ context
            
            # Blend global and per-arm predictions
            mu = (1.0 - blend) * mu_arm + blend * mu_global
            
            # Variance: combine per-arm and global uncertainty
            var_arm = context @ (self.inv_A[a] @ context)
            var = (1.0 - blend) ** 2 * var_arm + blend ** 2 * var_global
            std = np.sqrt(max(0.0, var))

            # UCB with confidence-weighted exploration
            # Reduce exploration bonus for arms with many samples
            count_factor = 1.0 / (1.0 + 0.1 * self.arm_counts[a])
            scores[a] = mu + alpha * std * count_factor
            self._last_mu[a] = mu

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self._deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error BEFORE updating
        pred = self._last_mu[arm]
        error = abs(reward - pred)
        self.error_buffer.append(error)

        # Adaptive forgetting factor
        lam = self._compute_lambda()

        # Update per-arm linear model with forgetting
        # A_new = lam * A_old + x x^T
        # b_new = lam * b_old + r * x
        A_old = self.A[arm]
        b_old = self.b[arm]
        
        A_new = lam * A_old + np.outer(context, context)
        b_new = lam * b_old + reward * context

        # Update inverse via Sherman-Morrison: (lam*A + xx^T)^{-1}
        # = (1/lam) * (A^{-1} - A^{-1} x x^T A^{-1} / (lam + x^T A^{-1} x))
        inv_A_old = self.inv_A[arm]
        Ax = inv_A_old @ context
        xAx = context @ Ax
        denom = lam + xAx
        
        if abs(denom) > 1e-10:
            inv_A_new = (1.0 / lam) * (inv_A_old - np.outer(Ax, Ax) / denom)
            # Symmetrize to reduce numerical drift
            inv_A_new = 0.5 * (inv_A_new + inv_A_new.T)
        else:
            # Fallback: recompute with regularization
            try:
                inv_A_new = np.linalg.inv(A_new + 1e-6 * np.eye(self.d))
            except np.linalg.LinAlgError:
                inv_A_new = np.eye(self.d)

        # Store updates
        self.A[arm] = A_new
        self.b[arm] = b_new
        self.inv_A[arm] = inv_A_new

        # Update shared global model with forgetting
        A_g_old = self.A_g
        b_g_old = self.b_g
        
        A_g_new = lam * A_g_old + np.outer(context, context)
        b_g_new = lam * b_g_old + reward * context

        inv_A_g_old = self.inv_A_g
        Ax_g = inv_A_g_old @ context
        xAx_g = context @ Ax_g
        denom_g = lam + xAx_g
        
        if abs(denom_g) > 1e-10:
            inv_A_g_new = (1.0 / lam) * (inv_A_g_old - np.outer(Ax_g, Ax_g) / denom_g)
            # Symmetrize to reduce numerical drift
            inv_A_g_new = 0.5 * (inv_A_g_new + inv_A_g_new.T)
        else:
            # Fallback: recompute with regularization
            try:
                inv_A_g_new = np.linalg.inv(A_g_new + 1e-6 * np.eye(self.d))
            except np.linalg.LinAlgError:
                inv_A_g_new = np.eye(self.d)

        # Store global updates
        self.A_g = A_g_new
        self.b_g = b_g_new
        self.inv_A_g = inv_A_g_new

        # Update arm count
        self.arm_counts[arm] += 1

# EVOLVE-BLOCK-END
