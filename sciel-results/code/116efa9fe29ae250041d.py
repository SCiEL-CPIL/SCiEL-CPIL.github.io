"""Non-stationary linear contextual bandit with global model and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Global linear model with arm indicators (one-hot)
    - Adaptive forgetting
    - UCB exploration
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_global = dim + n_arms
        self.start_time = time.time()
        self.deadline = 115.0

        # Global inverse covariance
        ridge = 0.1
        self.V = np.eye(self.d_global) * (1.0 + ridge)
        self.b = np.zeros(self.d_global)
        
        # Counts per arm for adaptive C
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.95
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        self.C = 1.5
        self.C_min = 0.2
        self.C_max = 4.5

        # Prediction error tracking
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.30
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        c = self.counts[arm]
        if c < 3:
            return self.base_gamma
        err_ema = self.pred_err_ema[arm]
        err_scale = min(err_ema, 6.0)
        sig = 1.0 / (1.0 + np.exp(-1.2 * (err_scale - 0.5)))
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * sig
        blend = 0.3 * (1.0 - sig)
        gamma = blend * self.base_gamma + (1.0 - blend) * gamma
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 6:
            return self.C * 1.4
        t = float(self.t)
        if t < 500.0:
            time_decay = np.exp(-t / 300.0)
            time_decay = max(0.4, time_decay)
        else:
            frac = min((t - 500.0) / 2000.0, 1.0)
            time_decay = 0.4 - frac * (0.4 - self.C_min / self.C)
            time_decay = max(self.C_min / self.C, time_decay)
        return self.C * time_decay

    def _make_context(self, context, arm):
        z = np.zeros(self.d_global)
        z[:self.d] = context
        z[self.d + arm] = 1.0
        return z

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            theta = self.V @ self.b
            scores = []
            for a in range(self.n):
                z = self._make_context(context, a)
                scores.append(z @ theta)
            return int(np.argmax(scores))

        theta = self.V @ self.b
        scores = []
        for a in range(self.n):
            z = self._make_context(context, a)
            mu = z @ theta
            c_val = self._adaptive_C(a)
            Vz = self.V @ z
            var = z @ Vz
            bonus = c_val * np.sqrt(max(var, 0.0))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1
        self.counts[arm] += 1

        z = self._make_context(context, arm)

        # Prediction error
        theta = self.V @ self.b
        pred = z @ theta
        err = reward - pred
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + self.err_ema_alpha * (err * err)

        gamma = self._adaptive_gamma(arm)

        # Update V
        Vz = self.V @ z
        zVz = z @ Vz
        denom = gamma + zVz

        if denom > 1e-12:
            self.V = (1.0 / gamma) * self.V - (1.0 / gamma) * np.outer(Vz, Vz) / denom
        else:
            self.V = np.eye(self.d_global) * (1.0 + 0.1)

        # Update b
        self.b = gamma * self.b + reward * z

        # Ridge stabilization
        ridge_step = 0.0008 / (1.0 + self.t * 0.01)
        if self.t % 10 == 0:
            self.V += ridge_step * np.eye(self.d_global)
# EVOLVE-BLOCK-END
