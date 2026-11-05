# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm Adaptive Discounted Recursive Least Squares (AD-RLS) for non-stationary linear bandits.

    Replaces the Kalman filter with a lighter recursive update using a dynamically
    adjusted forgetting factor per arm. The discount rate is inversely proportional
    to recent innovation variance to accelerate adaptation during high-volatility
    periods while maintaining stability during low-noise regimes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0

        # Per-arm state: theta (weight vector), P (inverse Hessian / covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Observation noise
        self._R = 0.1

        # Exploration parameters
        self.alpha = 1.2
        self._alpha_min = 0.3

        # Adaptive discounting parameters
        self._lambda_min = 0.90
        self._lambda_max = 0.995
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15

        # Reset mechanism
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

    def _adaptive_lambda(self, arm):
        """Compute adaptive forgetting factor based on innovation EMA.

        Higher innovation variance -> lower lambda (faster forgetting/adaptation).
        Lower innovation variance -> higher lambda (slower forgetting/stability).
        """
        if self.count[arm] < 3:
            return self._lambda_max

        ema_val = self._innov_ema[arm]
        # Map EMA to [0, 1] range using tanh
        scaled = np.tanh(ema_val / 0.4)
        # Inverse relationship: high variance -> low lambda
        lam = self._lambda_max - (self._lambda_max - self._lambda_min) * scaled
        return lam

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)

        if self._consec_large[arm] >= self._reset_threshold:
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm using UCB with RLS estimates."""
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ ctx
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)

            # Adaptive exploration: count-scaled and time-decayed
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))

            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Compute adaptive forgetting factor
        lam = self._adaptive_lambda(a)

        # Innovation
        y = r - self.theta[a] @ x

        # RLS Update
        # P_new = (1/lam) * P - (1/lam) * P x (R + x^T P x)^-1 x^T P
        P_x = self.P[a] @ x
        denom = self._R + x @ P_x
        if denom < 1e-12:
            denom = 1e-12

        # Gain vector
        K = P_x / denom

        # Update theta
        self.theta[a] = self.theta[a] + K * y

        # Update P using matrix inversion lemma with discounting
        # P_new = (1/lam) * (P - P x (R + x^T P x)^-1 x^T P)
        outer = np.outer(P_x, P_x)
        P_update = self.P[a] - outer / denom
        self.P[a] = (1.0 / lam) * P_update

        # Ensure symmetry
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track innovation for adaptive lambda
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
