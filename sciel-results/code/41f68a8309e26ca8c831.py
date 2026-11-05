"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        # Start with slight ridge for numerical stability
        ridge = 0.1
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.96
        self.gamma_min = 0.82
        self.gamma_max = 0.995
        self.C = 1.0  # exploration coefficient
        self.C_min = 0.2
        self.C_max = 4.0

        # For tracking prediction errors to adapt forgetting
        # Use exponential moving average for faster reaction
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.3  # EMA decay for error tracking

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction error (EMA)."""
        c = self.counts[arm]
        if c < 3:
            return self.base_gamma

        # Use EMA of squared prediction error to adjust gamma
        # Higher recent error -> faster forgetting (lower gamma)
        err_ema = self.pred_err_ema[arm]
        # Scale: map err_ema to gamma adjustment
        # err_ema ~ 0 -> gamma ~ base_gamma
        # err_ema ~ 1+ -> gamma drops significantly
        err_scale = min(err_ema, 4.0)
        # Smooth, monotonic mapping
        gamma = self.base_gamma - 0.08 * (1.0 - np.exp(-0.5 * err_scale))
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time with smooth decay."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 8:
            return self.C * 1.5
        # Smooth exponential decay of exploration over time
        # Faster decay than before to reduce unnecessary exploration
        time_decay = np.exp(-self.t / 800.0)
        time_decay = max(0.3, time_decay)
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        # Update EMA of squared error
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + self.err_ema_alpha * (err * err)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d) * (1.0 + 0.1)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic numerical stabilization: every 200 steps, add small ridge
        if self.t % 200 == 0:
            for a in range(self.n):
                self.V[a] += 0.001 * np.eye(self.d)
# EVOLVE-BLOCK-END
