import numpy as np
import time
from collections import deque

class Policy:
    """
    Global linear model with arm-specific bias terms.
    Structural change: single shared theta for context features,
    per-arm scalar biases for arm-specific reward offsets.
    Uses Sherman-Morrison for fast inverse updates and adaptive
    forgetting based on prediction errors.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = self._t0 + 115.0

        # Global linear model: A (d x d), b (d), inv_A (d x d)
        self.A_g = np.eye(dim)
        self.b_g = np.zeros(dim)
        self.inv_A_g = np.eye(dim)

        # Per-arm bias terms
        self.bias = np.zeros(n_arms)
        self.bias_counts = np.zeros(n_arms)

        # Adaptive forgetting
        self.base_lambda = 0.90
        self.min_lambda = 0.75
        self.max_lambda = 0.995
        self.error_buffer = deque(maxlen=40)

        # Adaptive UCB alpha
        self.base_alpha = 1.2
        self.min_alpha = 0.4
        self.max_alpha = 3.0

        # Round counter
        self._round = 0

        # Track last predicted mu for error computation
        self._last_mu = np.zeros(n_arms)

        # Arm pull counts for exploration decay
        self.arm_counts = np.zeros(n_arms)

    def _compute_lambda(self):
        """Adaptive forgetting factor from recent absolute errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = np.array(list(self.error_buffer))
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Robust scale
        if len(recent) >= 10:
            q1 = np.percentile(recent, 25)
            q3 = np.percentile(recent, 75)
            iqr = q3 - q1
            err_scale = max(0.05, iqr * 1.5)
        else:
            err_scale = 0.1

        normalized = (mean_err + 0.5 * std_err) / err_scale

        if normalized < 0.1:
            return self.max_lambda
        elif normalized > 2.0:
            return self.min_lambda
        else:
            t = (normalized - 0.1) / (2.0 - 0.1)
            t = t * t * (3.0 - 2.0 * t)  # smoothstep
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _compute_alpha(self):
        """Adaptive exploration parameter."""
        if len(self.error_buffer) < 10:
            return self.base_alpha

        recent = np.array(list(self.error_buffer))
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

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
        elif normalized > 1.8:
            return self.max_alpha
        else:
            t = (normalized - 0.15) / (1.8 - 0.15)
            t = t * t * (3.0 - 2.0 * t)
            return self.min_alpha + t * (self.max_alpha - self.min_alpha)

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        alpha = self._compute_alpha()

        # Global model prediction
        theta_g = self.inv_A_g @ self.b_g
        mu_global = theta_g @ context
        var_global = context @ (self.inv_A_g @ context)
        std_global = np.sqrt(max(0.0, var_global))

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Per-arm prediction: global linear + arm bias
            mu = mu_global + self.bias[a]

            # Uncertainty: global variance + bias uncertainty
            # Bias uncertainty decreases with pulls
            bias_var = 1.0 / (1.0 + 0.5 * self.bias_counts[a])
            total_var = var_global + bias_var
            std = np.sqrt(max(0.0, total_var))

            # Exploration decay for well-sampled arms
            count_factor = 1.0 / (1.0 + 0.15 * self.arm_counts[a])

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

        # Update global linear model with forgetting
        # A_new = lam * A_old + x x^T
        # b_new = lam * b_old + r * x
        A_old = self.A_g
        b_old = self.b_g

        A_new = lam * A_old + np.outer(context, context)
        b_new = lam * b_old + reward * context

        # Sherman-Morrison update for inverse
        inv_A_old = self.inv_A_g
        Ax = inv_A_old @ context
        xAx = context @ Ax
        denom = lam + xAx

        if abs(denom) > 1e-10:
            inv_A_new = (1.0 / lam) * (inv_A_old - np.outer(Ax, Ax) / denom)
            # Symmetrize
            inv_A_new = 0.5 * (inv_A_new + inv_A_new.T)
        else:
            try:
                inv_A_new = np.linalg.inv(A_new + 1e-6 * np.eye(self.d))
            except np.linalg.LinAlgError:
                inv_A_new = np.eye(self.d)

        self.A_g = A_new
        self.b_g = b_new
        self.inv_A_g = inv_A_new

        # Update per-arm bias with exponential moving average
        # Residual after removing global prediction
        theta_g = self.inv_A_g @ self.b_g
        pred_global = theta_g @ context
        residual = reward - pred_global

        # Exponential moving average for bias
        self.bias[arm] = lam * self.bias[arm] + (1.0 - lam) * residual
        self.bias_counts[arm] += 1
        self.arm_counts[arm] += 1
