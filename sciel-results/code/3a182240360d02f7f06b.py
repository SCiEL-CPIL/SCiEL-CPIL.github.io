# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Hierarchical Bayesian Linear Bandit.

    Decomposes reward into a shared global context model (slow forgetting,
    tracks long-term trends) plus arm-specific offsets (fast forgetting,
    tracks immediate regime shifts). This separates stable exploitation
    from rapid adaptation. Adaptive exploration scales with uncertainty
    and with detected prediction-error spikes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._deadline = time.time() + 110.0  # safety margin below 120s

        # Shared global model (slow forgetting)
        self.A_g = np.eye(dim)
        self.b_g = np.zeros(dim)
        self.g_g = 0.995  # slow: long-term trend

        # Arm-specific models (fast forgetting)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.g_a = 0.90   # fast: regime shifts

        # Per-arm running prediction error (EMA) for adaptive exploration
        self.err = [0.0 for _ in range(n_arms)]
        self.err_base = 1.0

        # Exploration base scale
        self.beta = 0.8

        self._t = 0

    def _inv(self, A):
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.eye(self.d)

    def select(self, context):
        x = np.asarray(context, dtype=float)

        # Global (slow) estimate
        inv_g = self._inv(self.A_g)
        theta_g = inv_g @ self.b_g
        mean_g = float(theta_g @ x)
        var_g = float(x @ inv_g @ x)

        # Per-arm (fast) estimate
        scores = []
        for a in range(self.n):
            inv = self._inv(self.A[a])
            theta = inv @ self.b[a]
            mean_a = float(theta @ x)
            var_a = float(x @ inv @ x)

            # Combined mean: global trend + arm offset
            mean = mean_g + mean_a

            # Adaptive exploration: scale with uncertainty and error spike
            err_ratio = self.err[a] / (self.err_base + 1e-9)
            boost = 1.0 + 0.6 * (err_ratio - 1.0)
            boost = max(0.5, min(2.5, boost))

            uncertainty = np.sqrt(var_g * 0.5 + var_a)
            scores.append(mean + self.beta * boost * uncertainty)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=float)
        r = float(reward)

        # Predicted mean for the chosen arm (before update) for error tracking
        inv_g = self._inv(self.A_g)
        pred_g = float((inv_g @ self.b_g) @ x)
        inv_a = self._inv(self.A[arm])
        pred_a = float((inv_a @ self.b[arm]) @ x)
        pred = pred_g + pred_a
        sq_err = (r - pred) ** 2

        # EMA of squared error (fast)
        self.err[arm] = 0.9 * self.err[arm] + 0.1 * sq_err
        # Slow baseline across arms
        self.err_base = 0.99 * self.err_base + 0.01 * sq_err

        # Update arm-specific (fast) model
        self.A[arm] = self.g_a * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.g_a * self.b[arm] + r * x

        # Update global (slow) model
        self.A_g = self.g_g * self.A_g + np.outer(x, x)
        self.b_g = self.g_g * self.b_g + r * x

        self._t += 1
# EVOLVE-BLOCK-END
