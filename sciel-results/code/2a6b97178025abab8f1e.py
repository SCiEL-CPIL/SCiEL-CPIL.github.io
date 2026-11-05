"""Combined approach: Global linear model with one-hot arm indicators, adaptive forgetting, and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit with:
    - Global linear model: context augmented with one-hot arm indicators
      to capture arm-specific biases in a single shared weight vector.
    - Discounted Least Squares with Sherman-Morrison updates (O(d^2) per step).
    - Adaptive forgetting rate based on prediction residuals.
    - UCB-style exploration with uncertainty bonus.
    - Deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.extended_dim = dim + n_arms  # context + one-hot arm indicators
        self.start_time = time.time()
        self.deadline = 135.0  # Safety margin before 150s kill

        # Ridge parameter for stability
        self.ridge = 0.1

        # Single global inverse covariance matrix
        # V = (sum gamma^t x_ext x_ext^T + lambda I)^{-1}
        self.V = (1.0 / self.ridge) * np.eye(self.extended_dim)
        self.b = np.zeros(self.extended_dim)

        # For adaptive forgetting
        self.prediction_errors = []
        self.max_err_window = 50

        # Exploration
        self.C = 1.5
        self.base_gamma = 0.94
        self.gamma_min = 0.80
        self.gamma_max = 0.99

        # Tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _make_extended_context(self, context, arm):
        """Augment context with one-hot arm indicator."""
        context = np.asarray(context, dtype=np.float64)
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate([context, one_hot])

    def _adaptive_gamma(self):
        """Adapt forgetting factor based on recent prediction errors."""
        if len(self.prediction_errors) < 10:
            return self.base_gamma

        recent_errs = self.prediction_errors[-10:]
        older_errs = self.prediction_errors[:-10]

        recent_mean = np.mean(recent_errs)
        older_mean = np.mean(older_errs)

        if older_mean < 1e-12:
            ratio = 1.0
        else:
            ratio = recent_mean / older_mean

        # Smooth nonlinear mapping
        log_ratio = np.log(max(ratio, 1e-6))
        adjustment = -2.0 * log_ratio
        gamma = self.base_gamma + adjustment

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def select(self, context):
        """Select an arm using UCB with the global model."""
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                ext_ctx = self._make_extended_context(context, a)
                mu = self.V @ self.b
                scores.append(mu @ ext_ctx)
            return int(np.argmax(scores))

        # UCB with uncertainty bonus
        scores = []
        for a in range(self.n):
            ext_ctx = self._make_extended_context(context, a)
            theta = self.V @ self.b
            mu = theta @ ext_ctx
            # Uncertainty: sqrt(x^T V x)
            Vx = self.V @ ext_ctx
            bonus = self.C * np.sqrt(ext_ctx @ Vx)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global model using Sherman-Morrison with forgetting."""
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Compute prediction error before updating
        ext_ctx = self._make_extended_context(context, arm)
        theta = self.V @ self.b
        pred = theta @ ext_ctx
        err = abs(reward - pred)
        self.prediction_errors.append(err)
        if len(self.prediction_errors) > self.max_err_window:
            self.prediction_errors.pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma()

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V
        Vx = V_old @ ext_ctx
        xVx = ext_ctx @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V = (1.0 / self.ridge) * np.eye(self.extended_dim)

        # Update b with forgetting
        self.b = gamma * self.b + reward * ext_ctx
# EVOLVE-BLOCK-END
