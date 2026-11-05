"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least squares (ridge regression with exponential forgetting)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 135.0  # safety margin before 150s kill

        # Ridge parameter for stability
        self.ridge = 0.10

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        self.V = [(1.0 / self.ridge) * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # For adaptive forgetting
        self.prediction_errors = [[] for _ in range(n_arms)]
        self.max_err_window = 80

        # Exploration
        self.C = 1.2
        self.base_gamma = 0.93
        self.gamma_min = 0.70
        self.gamma_max = 0.98

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction errors with smoother response."""
        c = self.counts[arm]
        if c < 8:
            return self.base_gamma

        # Use recent prediction errors to detect non-stationarity
        recent_errs = self.prediction_errors[arm]
        if len(recent_errs) < 15:
            return self.base_gamma

        # Compare recent errors to earlier ones using medians for robustness
        n = len(recent_errs)
        recent_window = min(15, n // 2)
        older_window = min(15, n - recent_window)
        
        if older_window < 5:
            return self.base_gamma
            
        recent_mean = np.median(recent_errs[-recent_window:])
        older_mean = np.median(recent_errs[:older_window])

        if older_mean < 1e-10:
            ratio = 1.0
        else:
            ratio = recent_mean / older_mean

        # Smooth nonlinear mapping: use log-ratio for smoother response
        log_ratio = np.log(max(ratio, 1e-6))
        # Map log_ratio to gamma adjustment with tanh saturation for smoothness
        # log_ratio > 0 means errors increasing -> need faster forgetting (lower gamma)
        # log_ratio < 0 means errors decreasing -> can use slower forgetting (higher gamma)
        adjustment = -3.0 * np.tanh(log_ratio)
        gamma = self.base_gamma + adjustment

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient with smooth decay based on count."""
        c = self.counts[arm]
        if c < 1:
            return self.C * 4.0
        # Smooth decay: C * (1 + 2.0 / (1 + c/5))
        return self.C * (1.0 + 2.0 / (1.0 + c / 5.0))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error before updating
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        innovation = reward - pred
        err = abs(innovation)
        
        # Huber-style robustification: cap the effective reward contribution
        # for very large innovations to reduce outlier sensitivity
        V_old = self.V[arm]
        Vx = V_old @ context
        sigma_sq = max(context @ Vx, 1e-8)  # predicted variance
        sigma = np.sqrt(sigma_sq)
        
        # Huber threshold: 1.5 * sigma
        huber_threshold = 1.5 * sigma
        if err > huber_threshold:
            # Use Huber loss: linear beyond threshold
            effective_reward = pred + np.sign(innovation) * huber_threshold
        else:
            effective_reward = reward

        self.prediction_errors[arm].append(err)
        if len(self.prediction_errors[arm]) > self.max_err_window:
            self.prediction_errors[arm].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = (1.0 / self.ridge) * np.eye(self.d)

        # Update b with forgetting using effective reward for robustness
        self.b[arm] = gamma * self.b[arm] + effective_reward * context
# EVOLVE-BLOCK-END
