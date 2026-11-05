"""Differential Linear Bandit with global trend subtraction."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Differential Linear Bandit:
    - Maintains a global linear trend estimator (context -> reward) using a forgetting factor.
    - Subtracts the predicted trend from observed rewards to get de-trended residuals.
    - Applies LinUCB on residuals per arm to isolate arm-specific signal.
    - Uses a tanh-saturated EMA for adaptive exploration scaling.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm LinUCB state
        self.alpha = 1.0  # exploration constant
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Global trend estimator (single linear model for all arms)
        self.lam_trend = 0.99  # forgetting factor for trend
        self.A_trend = np.eye(dim) * 1.0
        self.b_trend = np.zeros(dim)

        # Adaptive exploration via tanh-saturated EMA of residual magnitude
        self.residual_ema = 1.0  # initial estimate of typical residual scale
        self.ema_beta = 0.95

        # Track number of updates per arm for count-based alpha decay
        self.counts = np.zeros(n_arms, dtype=np.int64)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        scores = []
        for a in range(self.n):
            inv_A = np.linalg.inv(self.A[a])
            theta = inv_A @ self.b[a]
            mu = theta @ context

            # Uncertainty term with adaptive scaling
            # Use tanh-saturated EMA to bound the exploration bonus
            var = context @ inv_A @ context
            # Scale uncertainty by current residual EMA, bounded via tanh
            scale = np.tanh(self.residual_ema)
            bonus = self.alpha * scale * np.sqrt(var)

            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update global trend estimator
        self.A_trend = self.lam_trend * self.A_trend + np.outer(context, context)
        self.b_trend = self.lam_trend * self.b_trend + reward * context

        # Predict trend
        inv_A_trend = np.linalg.inv(self.A_trend)
        theta_trend = inv_A_trend @ self.b_trend
        trend_pred = theta_trend @ context

        # De-trended residual
        residual = reward - trend_pred

        # Update residual EMA (tanh-saturated for stability)
        self.residual_ema = self.ema_beta * self.residual_ema + (1 - self.ema_beta) * abs(residual)

        # Update per-arm LinUCB with residual
        self.A[arm] = 0.995 * self.A[arm] + np.outer(context, context)
        self.b[arm] = 0.995 * self.b[arm] + residual * context

        # Increment count
        self.counts[arm] += 1

        # Adaptive alpha: reduce exploration for well-visited arms
        # alpha decays as 1 / (1 + 0.1 * count) but bounded below
        self.alpha = max(0.3, 1.0 / (1.0 + 0.1 * self.counts[arm]))
# EVOLVE-BLOCK-END
