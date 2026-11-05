# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm Kalman filter with Huber-robust innovation handling and adaptive process noise.
    
    Improvements over baseline:
    1. Huber loss weighting on innovations to reduce sensitivity to outliers/regime shifts
    2. Adaptive process noise based on EMA of squared prediction errors
    3. Adaptive UCB coefficient that decays with time but remains responsive
    4. Joseph form covariance update for numerical stability
    5. Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.time_limit = 140.0  # safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (scalar, scaled to identity)
        self.Q_scalar = [1e-3 for _ in range(n_arms)]
        self.base_Q = 1e-3

        # EMA of squared prediction error per arm
        self.mse_ema = [0.0 for _ in range(n_arms)]
        self.error_count = [0 for _ in range(n_arms)]

        # Huber threshold parameter
        self.huber_delta = 1.5

        # Adaptive UCB coefficient
        self.beta_base = 2.0
        self.beta_decay = 0.001

    def _get_adaptive_Q(self, arm):
        """Get adaptive process noise matrix for an arm."""
        # Scale process noise based on recent prediction error magnitude
        if self.error_count[arm] >= 5:
            # Use sqrt of MSE as a scale indicator
            err_scale = np.sqrt(self.mse_ema[arm])
            # Smoothly increase Q when errors are large, cap to avoid instability
            adaptive_component = 0.05 * min(err_scale, 3.0)
            self.Q_scalar[arm] = self.base_Q + adaptive_component
        return self.Q_scalar[arm] * np.eye(self.d)

    def _huber_weight(self, innovation, S):
        """Compute Huber loss weight for robust update.
        
        Returns a weight in (0, 1] that downweights large innovations.
        """
        if S <= 0:
            return 1.0
        abs_innov = abs(innovation)
        threshold = self.huber_delta * np.sqrt(S)
        if abs_innov <= threshold:
            return 1.0
        else:
            # Linear decay: weight = threshold / |innovation|
            return threshold / abs_innov

    def _adaptive_beta(self):
        """Adaptive UCB coefficient that decays over time."""
        # Decay slowly but keep minimum exploration
        return max(self.beta_base * np.exp(-self.beta_decay * self.t), 0.3)

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        beta = self._adaptive_beta()
        scores = []

        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            Q_a = self._get_adaptive_Q(a)
            P_pred = self.P[a] + Q_a

            # Prediction
            pred_reward = self.theta[a] @ context

            # UCB score: mean + beta * uncertainty
            uncertainty = np.sqrt(max(context @ P_pred @ context, 0.0))
            score = pred_reward + beta * uncertainty

            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        theta = self.theta[arm]
        P = self.P[arm]

        # Predict step
        Q_a = self._get_adaptive_Q(arm)
        P_pred = P + Q_a

        # Prediction
        pred_reward = theta @ context

        # Innovation
        innovation = reward - pred_reward

        # Update MSE EMA
        sq_err = innovation * innovation
        if self.error_count[arm] == 0:
            self.mse_ema[arm] = sq_err
        else:
            alpha = 0.1
            self.mse_ema[arm] = (1 - alpha) * self.mse_ema[arm] + alpha * sq_err
        self.error_count[arm] += 1

        # Innovation variance
        S = context @ P_pred @ context + 1.0

        # Huber robustness weight
        w = self._huber_weight(innovation, S)

        # Effective innovation and noise
        # With Huber loss, we scale the innovation and adjust the noise
        S_eff = S / w
        K = P_pred @ context / S_eff

        # Update theta with weighted innovation
        self.theta[arm] = theta + K * (innovation * w)

        # Joseph form for numerical stability
        # Effective observation noise reflects Huber adjustment
        R_eff = 1.0 / w
        I_K_ctx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_K_ctx @ P_pred @ I_K_ctx.T + np.outer(K, K) * R_eff

        # Clamp covariance to prevent numerical drift
        self.P[arm] = np.maximum(self.P[arm], 1e-8 * np.eye(self.d))

        self.count[arm] += 1
# EVOLVE-BLOCK-END
