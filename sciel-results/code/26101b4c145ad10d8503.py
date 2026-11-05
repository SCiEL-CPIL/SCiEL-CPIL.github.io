"""Non-stationary linear contextual bandit with refined adaptive DLS and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least squares with adaptive ridge
    - Smooth EMA-based adaptive forgetting rate
    - Time-decaying UCB exploration
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Per-arm parameters
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Smooth adaptive forgetting: EMA of squared prediction errors
        self.err_ema = [0.0] * n_arms
        self.err_ema_beta = 0.9  # slower EMA for stability
        self.err_ema_beta_fast = 0.7  # faster EMA for responsiveness

        # Exploration
        self.C = 2.0
        self.base_gamma = 0.96
        self.gamma_min = 0.89
        self.gamma_max = 0.995

        # Total rounds seen (for time-decay)
        self.total_rounds = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Smooth adaptive forgetting using EMA ratio of error rates."""
        c = self.counts[arm]
        if c < 8:
            return self.base_gamma

        # Use two EMAs: fast and slow, compare them
        fast = self.err_ema[arm]  # This is actually the fast one
        # We need a second slower EMA - store both
        if not hasattr(self, 'err_ema_slow'):
            self.err_ema_slow = [0.0] * self.n

        slow = self.err_ema_slow[arm]

        if slow < 1e-12:
            ratio = 1.0
        else:
            ratio = fast / slow

        # Smoothly map ratio to gamma using tanh
        # ratio > 1 means errors increasing -> want smaller gamma (faster forgetting)
        # ratio < 1 means errors decreasing -> want larger gamma (slower forgetting)
        # tanh gives smooth -1 to 1 mapping
        t = np.tanh(0.8 * (ratio - 1.0))
        # Map t from [-1,1] to [gamma_min, gamma_max]
        gamma = self.base_gamma + t * (self.gamma_max - self.base_gamma)
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))

        return gamma

    def _adaptive_C(self, arm):
        """Time-decaying exploration with arm-specific count adjustment."""
        c = self.counts[arm]
        # Base decay: explore more early, less later
        time_decay = 1.0 / (1.0 + self.total_rounds * 0.005)
        # Count-based: explore less for well-visited arms
        count_factor = 1.0 / (1.0 + c * 0.1)
        return self.C * time_decay * count_factor * 2.0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = np.array([self.V[a] @ self.b[a] @ context for a in range(self.n)])
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            c_val = self._adaptive_C(a)
            # Uncertainty: sqrt(x^T V x)
            Vx = self.V[a] @ context
            uncertainty = np.sqrt(max(0.0, context @ Vx))
            bonus = c_val * uncertainty
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.total_rounds += 1

        # Compute prediction error before updating
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err_sq = (reward - pred) ** 2

        # Update EMAs
        if not hasattr(self, 'err_ema_slow'):
            self.err_ema_slow = [0.0] * self.n
        self.err_ema[arm] = self.err_ema_beta_fast * self.err_ema[arm] + (1 - self.err_ema_beta_fast) * err_sq
        self.err_ema_slow[arm] = self.err_ema_beta * self.err_ema_slow[arm] + (1 - self.err_ema_beta) * err_sq

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
