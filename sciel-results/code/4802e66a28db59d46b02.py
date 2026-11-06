"""Adaptive linear bandit policy for non-stationary settings."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Hybrid linear bandit policy: global linear model + arm-specific biases.
    
    Model: reward = x^T theta_global + bias_a + noise
    - Single shared linear parameter vector theta_global for context features.
    - Per-arm scalar biases for arm-specific offsets.
    - Adaptive forgetting factor reacts to prediction errors.
    - Adaptive UCB alpha based on recent error magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Global linear model
        # A_g: dim x dim matrix
        # b_g: dim vector
        # inv_A_g: dim x dim matrix (inverse of A_g)
        self.A_g = np.eye(dim)
        self.b_g = np.zeros(dim)
        self.inv_A_g = np.eye(dim)

        # Arm-specific biases
        self.bias = np.zeros(n_arms)
        self.bias_var = np.ones(n_arms)  # Variance of bias estimates

        # Count of observations per arm for bias estimation
        self.arm_counts = np.zeros(n_arms)

        # Adaptive forgetting
        self.base_lambda = 0.93
        self.min_lambda = 0.80
        self.max_lambda = 0.99
        self.error_buffer = deque(maxlen=60)

        # Adaptive UCB
        self.base_alpha = 1.2
        self.min_alpha = 0.4
        self.max_alpha = 3.0

        # Track predictions for error computation
        self._last_mu = np.zeros(n_arms)

        # Time guard
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0

        # Round counter
        self._round = 0

    def _compute_lambda(self):
        """Adaptive forgetting factor based on recent prediction errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Robust error scale using IQR
        if len(recent) >= 10:
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + std_err * 0.5) / err_scale

        if normalized < 0.1:
            return self.max_lambda
        elif normalized > 2.5:
            return self.min_lambda
        else:
            # Smooth interpolation with smoothstep
            t = (normalized - 0.1) / (2.5 - 0.1)
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

        if normalized < 0.15:
            return self.min_alpha
        elif normalized > 2.0:
            return self.max_alpha
        else:
            t = (normalized - 0.15) / (2.0 - 0.15)
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

        # Compute global theta
        theta_g = self.inv_A_g @ self.b_g

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Prediction: x^T theta_g + bias_a
            mu = theta_g @ context + self.bias[a]

            # Variance from global model: x^T inv_A_g x
            var_global = context @ (self.inv_A_g @ context)
            # Variance from bias estimate
            var_bias = self.bias_var[a]
            total_var = var_global + var_bias
            std = np.sqrt(max(0.0, total_var))

            # UCB with decay on std for very confident arms
            if std < 0.1:
                effective_std = std * 0.5
            else:
                effective_std = std

            scores[a] = mu + alpha * effective_std
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

        # Update global linear model with forgetting
        # A_g_new = lam * A_g_old + x x^T
        # b_g_new = lam * b_g_old + r * x
        A_g_old = self.A_g
        b_g_old = self.b_g

        A_g_new = lam * A_g_old + np.outer(context, context)
        b_g_new = lam * b_g_old + reward * context

        # Update inverse via Sherman-Morrison
        inv_A_g_old = self.inv_A_g
        Ax = inv_A_g_old @ context
        xAx = context @ Ax
        denom = lam + xAx

        if abs(denom) > 1e-10:
            inv_A_g_new = (1.0 / lam) * (inv_A_g_old - np.outer(Ax, Ax) / denom)
            # Symmetrize to reduce numerical drift
            inv_A_g_new = 0.5 * (inv_A_g_new + inv_A_g_new.T)
        else:
            # Fallback: recompute with regularization
            try:
                inv_A_g_new = np.linalg.inv(A_g_new + 1e-6 * np.eye(self.d))
            except np.linalg.LinAlgError:
                inv_A_g_new = np.eye(self.d)

        # Store global model updates
        self.A_g = A_g_new
        self.b_g = b_g_new
        self.inv_A_g = inv_A_g_new

        # Update arm-specific bias with forgetting
        # The residual for this arm is reward - x^T theta_g
        # But since we're updating theta_g simultaneously, we use a simple
        # exponentially weighted moving average for the bias
        # bias_new = lam * bias_old + (1 - lam) * residual
        # where residual approximates the arm-specific offset

        # Compute the current global prediction (before bias)
        # We use the OLD theta_g for computing the residual to avoid
        # circular dependency
        theta_g_old = inv_A_g_old @ b_g_old
        global_pred = theta_g_old @ context
        residual = reward - global_pred

        # Exponentially weighted bias update
        self.bias[arm] = lam * self.bias[arm] + (1.0 - lam) * residual

        # Update bias variance: var_new = lam^2 * var_old + (1-lam)^2 * sigma^2
        # We approximate sigma^2 as the recent error variance
        if len(self.error_buffer) > 5:
            recent_errs = list(self.error_buffer)
            sigma2 = np.var(recent_errs)
        else:
            sigma2 = 1.0

        self.bias_var[arm] = lam * lam * self.bias_var[arm] + (1.0 - lam) ** 2 * sigma2

        # Update arm count
        self.arm_counts[arm] += 1

# EVOLVE-BLOCK-END
