"""Non-stationary linear contextual bandit with per-arm Kalman filter and context-scaled process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-arm Kalman filter with context-scaled process noise for non-stationary linear bandits.
    - Each arm maintains a Kalman filter state (theta, P) for its weight vector.
    - Process noise Q is scaled by context norm to adapt to drift magnitude.
    - UCB exploration with adaptive coefficient.
    - Deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s evaluator limit

        # Kalman filter states per arm: theta (mean), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initial covariance: large uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Kalman filter parameters
        self.R = 1.0  # observation noise variance
        self.Q_base = 0.01  # base process noise variance
        self.Q_lambda = 0.05  # scaling factor for context-norm-dependent process noise

        # Exploration parameters
        self.C = 1.5
        self.C_min = 0.3
        self.C_max = 5.0

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _context_scaled_Q(self, context):
        """Compute process noise matrix scaled by context norm."""
        ctx_norm_sq = float(context @ context)
        q_scalar = self.Q_base + self.Q_lambda * ctx_norm_sq
        return np.eye(self.d) * q_scalar

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 2.0
        # Decay exploration over time
        time_decay = max(0.4, 1.0 - 0.15 * (self.t / 500.0))
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                mu = self.theta[a] @ context
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # UCB bonus using Kalman covariance
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(max(0.0, context @ self.P[a] @ context))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1
        self.counts[arm] += 1

        # Kalman filter update for the selected arm
        theta = self.theta[arm]
        P = self.P[arm]

        # Prediction step: inflate covariance with process noise
        Q = self._context_scaled_Q(context)
        P_pred = P + Q

        # Update step: Kalman gain
        # y = reward - theta @ context (innovation)
        innovation = reward - theta @ context

        # S = H P H^T + R, where H = context (1 x d)
        S = context @ P_pred @ context + self.R

        if S < 1e-12:
            return

        # Kalman gain K = P H^T S^{-1}
        K = P_pred @ context / S

        # Update theta
        self.theta[arm] = theta + K * innovation

        # Update P (Joseph form for numerical stability)
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R
        # Symmetrize
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

        # Periodic numerical stabilization
        if self.t % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                P = (P + P.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(P)
                eigvals = np.clip(eigvals, 1e-8, 100.0)
                self.P[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
