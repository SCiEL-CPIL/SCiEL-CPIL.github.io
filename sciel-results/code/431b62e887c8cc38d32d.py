"""Non-stationary linear contextual bandit with adaptive Kalman filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with adaptive process noise and UCB exploration.
    
    The state for each arm is a weight vector theta. The dynamics are modeled
    as a random walk: theta_t = theta_{t-1} + w_t, w_t ~ N(0, Q_t).
    The process noise Q_t is adapted based on an EMA of the squared innovation
    (prediction error), allowing the filter to react to non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s kill

        # Kalman filter state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initial high uncertainty to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.q0 = 0.02          # Base process noise
        self.q_max = 5.0        # Cap on adaptive process noise to prevent instability
        self.innov_ema = [0.0] * n_arms  # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.3      # Fast adaptation rate for EMA
        self.innov_base = 1.0   # Baseline threshold for innovation

        # UCB exploration parameters
        self.u_scale = 1.5      # Base UCB multiplier
        self.count_scale = 0.4  # Dampening factor for bonus based on count
        self.warmup_steps = 30  # Steps to boost exploration early
        self.jitter = 1e-6      # Numerical stabilizer
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov):
        """
        Compute adaptive process noise scalar based on innovation.
        High innovation (surprise) leads to larger Q (faster forgetting).
        """
        c = self.count[arm]
        if c < 3:
            # Early phase: larger Q to help initial convergence
            return self.q0 * 2.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        # Scale Q based on how much the innovation exceeds the baseline
        excess = max(0.0, innov_ema - self.innov_base)
        # Exponential mapping for smooth, bounded scaling
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        q = min(self.q_max, self.q0 * q_scale)
        return q

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """Kalman update step: rank-1 update with innovation."""
        P = self.P[a]
        theta = self.theta[a]
        
        # Predicted reward
        pred = float(theta @ ctx)
        # Innovation
        innov = reward - pred
        
        # Innovation variance: S = x^T P x + R (assuming R=1 or included in jitter/q)
        # For linear bandits with Gaussian noise N(0, sigma^2), if we don't know sigma,
        # we often assume sigma^2=1 or adapt it. Here we assume R=1.
        R = 1.0
        var = float(ctx @ P @ ctx) + R + self.jitter
        
        # Kalman gain: K = P x / S
        K = (P @ ctx) / var
        
        # Update mean: theta <- theta + K * innov
        self.theta[a] = theta + (K * innov)
        
        # Joseph-form covariance update for numerical stability:
        # P <- (I - K x^T) P (I - K x^T)^T + K R K^T
        IK = np.eye(self.d) - np.outer(K, ctx)
        # Simplified Joseph form for rank-1 update with R=1:
        # P <- IK P IK^T + K K^T
        # However, the standard form P <- (I-KH)P is often sufficient if K is accurate.
        # Let's use the standard form for speed, with symmetrization.
        self.P[a] = IK @ P
        
        # Ensure symmetry and positive semi-definite
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        
        # Clip eigenvalues to ensure stability
        # Periodic full eigen-decomposition is expensive, so we do a cheap check:
        # If diagonal elements become negative or too large, reset/clip.
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            # Fallback: clip diagonal
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))
            
        self.count[a] += 1
        return innov

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            # Decay exploration slightly over time, but keep a floor
            # to handle non-stationarity
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest current mean estimate (no further exploration).
        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            
            # Uncertainty bonus
            var = float(ctx @ self.P[a] @ ctx)
            std = float(np.sqrt(max(var, 0.0)))
            
            c = self.count[a]
            # Dampen bonus as we get more data for this arm
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            
            # Boost for unseen arms
            if c == 0:
                bonus += 2.0  # Strong incentive to explore new arms
            
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovation before predict to drive adaptive Q
        pred = float(self.theta[arm] @ ctx)
        innov = reward - pred

        # Adaptive process noise
        q = self._adaptive_q(arm, innov)

        # Kalman predict step
        self._predict(arm, q)

        # Kalman update step
        self._update(arm, ctx, reward)

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                # Ensure symmetry
                P = 0.5 * (P + P.T)
                # Check for NaNs or infinities
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
# EVOLVE-BLOCK-END
