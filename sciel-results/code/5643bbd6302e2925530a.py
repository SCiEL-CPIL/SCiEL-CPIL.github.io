"""Per-Arm Adaptive Discounted RLS for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-Arm Adaptive Discounted Recursive Least Squares (AD-RLS) policy.
    - Uses a Kalman filter formulation with Joseph-form covariance update
    - Adaptive process noise (Q) driven by innovation variance
    - UCB exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0

        # Per-arm state: P (covariance), theta (mean), count
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Innovation tracking for adaptive process noise
        self.innovations = [[] for _ in range(n_arms)]
        self.max_innov_window = 30

        # Base parameters
        self.R = 1.0  # Measurement noise (assumed)
        self.Q_base = 0.01  # Base process noise
        self.Q_min = 0.005
        self.Q_max = 0.1
        self.C = 2.0  # Exploration coefficient

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_Q(self, arm):
        """Adapt process noise based on recent innovation variance."""
        c = self.counts[arm]
        if c < 5:
            return self.Q_base

        inno = self.innovations[arm]
        if len(inno) < 10:
            return self.Q_base

        # Compute recent innovation variance
        recent = np.array(inno[-10:])
        older = np.array(inno[:max(1, len(inno) - 10)])
        
        recent_var = np.var(recent)
        older_var = np.var(older) if len(older) > 1 else recent_var

        # If innovations are increasing, increase Q (more adaptation)
        if older_var < 1e-12:
            ratio = 1.0
        else:
            ratio = recent_var / older_var

        Q = self.Q_base
        if ratio > 1.5:
            # High volatility: increase process noise
            Q = min(self.Q_max, self.Q_base * ratio)
        elif ratio < 0.5:
            # Low volatility: decrease process noise for stability
            Q = max(self.Q_min, self.Q_base * ratio)

        return Q

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient."""
        c = self.counts[arm]
        if c < 5:
            return self.C * 2.0
        if c < 15:
            return self.C * 1.3
        return self.C

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.P[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.counts[arm] += 1

        # Predict step: propagate covariance with adaptive process noise
        Q = self._adaptive_Q(arm)
        P_pred = self.P[arm] + Q * np.eye(self.d)
        theta_pred = self.theta[arm]

        # Update step: Kalman gain and Joseph-form update
        S = context @ P_pred @ context + self.R  # Innovation variance
        if S < 1e-12:
            S = 1e-12
        K = (P_pred @ context) / S  # Kalman gain

        # Innovation
        innovation = reward - (theta_pred @ context)
        self.innovations[arm].append(innovation)
        if len(self.innovations[arm]) > self.max_innov_window:
            self.innovations[arm].pop(0)

        # Update mean
        self.theta[arm] = theta_pred + K * innovation

        # Joseph-form covariance update for numerical stability
        I_Kx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_Kx @ P_pred @ I_Kx.T + self.R * np.outer(K, K)

        # Symmetrize to prevent drift
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
# EVOLVE-BLOCK-END
