"""Bootstrap spawn: direct approach for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 138.0  # safety margin before 150s kill

        # Per-arm Kalman-like parameters:
        # P = inverse covariance (P = V in DLS), theta = mean estimate
        self.P = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # For adaptive forgetting / process noise
        self.prediction_errors = [[] for _ in range(n_arms)]
        self.max_err_window = 60
        self.residual_ratios = [[] for _ in range(n_arms)]  # for smooth non-stationarity detection

        # Exploration
        self.C = 2.2
        self.base_gamma = 0.935
        self.gamma_min = 0.80
        self.gamma_max = 0.988

        # Process noise scale (Q) base
        self.Q_base = 1e-4

        # Track last few contexts for possible restarts
        self.recent_contexts = []
        self.max_recent = 50

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction errors (smooth)."""
        c = self.counts[arm]
        if c < 8:
            return self.base_gamma

        recent_errs = self.prediction_errors[arm]
        if len(recent_errs) < 8:
            return self.base_gamma

        n = len(recent_errs)
        # Use a slightly wider window for stability
        recent_mean = np.mean(recent_errs[-12:])
        older_mean = np.mean(recent_errs[:max(1, n - 12)])

        if older_mean < 1e-10:
            ratio = 1.0
        else:
            ratio = recent_mean / older_mean

        # Smooth soft-min/soft-max surrogate: use a soft version of max(ratio, 1)
        # to avoid the kink at ratio=1. Use log-sum-exp style smoothing.
        # soft_max(ratio, 1) = log(exp(ratio) + exp(1)) - log(2)  -- but ratio can be large
        # Instead, use a smooth sigmoid-based blend:
        # When ratio > 1, we want to increase the "drift signal"
        # Use: drift_signal = 1 / (1 + exp(-2.0 * (ratio - 1.0)))  -- smooth step at ratio=1
        drift_signal = 1.0 / (1.0 + np.exp(-2.0 * (ratio - 1.0)))
        # drift_signal in [0,1], ~0.5 at ratio=1, ->1 as ratio->inf
        # Scale the gamma reduction by this smooth signal
        max_reduction = self.base_gamma - self.gamma_min
        gamma = self.base_gamma - drift_signal * max_reduction * 0.8

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient based on uncertainty and count (smooth)."""
        c = self.counts[arm]
        if c < 1:
            return self.C * 4.0
        # Smooth decay: C * (1 + 2.0 / sqrt(c + 1))
        return self.C * (1.0 + 2.0 / np.sqrt(c + 1.0))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        # UCB with adaptive exploration using explicit P (covariance)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            c_val = self._adaptive_C(a)
            # Uncertainty: sqrt(x^T P x)
            Px = self.P[a] @ context
            bonus = c_val * np.sqrt(max(0.0, context @ Px))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error before updating
        pred = self.theta[arm] @ context
        err = abs(reward - pred)
        self.prediction_errors[arm].append(err)
        if len(self.prediction_errors[arm]) > self.max_err_window:
            self.prediction_errors[arm].pop(0)

        # Adaptive forgetting (gamma) and process noise (Q)
        gamma = self._adaptive_gamma(arm)

        # Adaptive process noise: scale Q by recent residual growth
        c = self.counts[arm]
        if c > 10:
            recent_errs = self.prediction_errors[arm]
            recent_mean = np.mean(recent_errs[-10:])
            older_mean = np.mean(recent_errs[:max(1, len(recent_errs) - 10)])
            if older_mean > 1e-10:
                ratio = recent_mean / older_mean
                # Smooth scaling: Q grows when ratio > 1
                q_scale = max(1.0, ratio ** 0.7)
            else:
                q_scale = 1.0
        else:
            q_scale = 1.0

        Q = self.Q_base * q_scale * np.eye(self.d)

        # Kalman-like update:
        # 1. Predict: P_pred = P / gamma + Q  (forgetting + process noise)
        P = self.P[arm] / gamma + Q
        # 2. Update: K = P x / (1 + x^T P x)
        Px = P @ context
        denom = 1.0 + context @ Px
        if denom > 1e-12:
            K = Px / denom
            # theta = theta + K * (reward - theta^T x)
            innovation = reward - self.theta[arm] @ context
            self.theta[arm] = self.theta[arm] + K * innovation
            # P = (I - K x^T) P
            self.P[arm] = P - np.outer(K, context) @ P
        else:
            self.P[arm] = np.eye(self.d)
# EVOLVE-BLOCK-END
