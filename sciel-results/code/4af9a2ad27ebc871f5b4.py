"""Non-stationary linear contextual bandit with a joint Kalman filter + adaptive process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Joint Kalman-filter linear bandit with adaptive process noise."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self._t0 = time.perf_counter()
        self._deadline = 150.0 - 6.0  # safety margin before the 150s wall-clock kill

        # Initial beliefs: zero mean, large uncertainty
        self.theta = np.zeros((n_arms, dim))
        self.P = np.array([np.eye(dim) * 100.0 for _ in range(n_arms)])

        # Hyperparameters
        self.base_q = 0.05          # baseline process-noise scale
        self.spike_gain = 0.5       # how much residual spikes inflate Q
        self.coef = 0.02            # coherence pull toward the cross-arm mean
        self.bonus = 0.7            # exploration coefficient
        self.rng = np.random.default_rng(0)

    def _near_deadline(self):
        return (time.perf_counter() - self._t0) > self._deadline

    def select(self, context):
        context = np.asarray(context, dtype=float)
        self.t += 1

        # Once near the deadline, stop learning and just exploit (safe, no crash).
        if self._near_deadline():
            mu = self.theta @ context
            return int(np.argmax(mu))

        # Prediction step: inflate covariance with adaptive process noise
        for a in range(self.n):
            self.P[a] = self.P[a] + self._Q(a, context)

        # Shared coherence prior: gently pull each arm toward the cross-arm mean
        theta_mean = self.theta.mean(axis=0)
        for a in range(self.n):
            self.theta[a] = self.theta[a] + self.coef * (theta_mean - self.theta[a])

        # UCB scoring
        mu = self.theta @ context
        sigma = np.sqrt(np.maximum(np.einsum('ad,ad->a', context, self.P @ context), 1e-12))
        decay = 1.0 / np.sqrt(1.0 + 0.001 * self.t)
        scores = mu + self.bonus * decay * sigma
        return int(np.argmax(scores))

    def _Q(self, a, context):
        """Adaptive process-noise matrix for arm a."""
        # Residual of the last observation for this arm (stored on P update path)
        resid = getattr(self, '_resid', None)
        if resid is not None:
            r = resid[a]
            # standardized spike magnitude
            scale = 1.0 + self.spike_gain * min(abs(r), 5.0)
        else:
            scale = 1.0
        return np.eye(self.d) * (self.base_q * scale)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        reward = float(reward)
        a = int(arm)
        self._resid = np.zeros(self.n)  # will be filled below for next predict

        if self._near_deadline():
            # Keep state finite but skip heavy computation.
            self._resid[a] = reward - self.theta[a] @ context
            return

        # Innovation
        pred = self.theta[a] @ context
        resid = reward - pred
        self._resid[a] = resid

        # Kalman gain
        Pxx = self.P[a] @ context
        S = Pxx @ context + 1.0
        K = Pxx / max(S, 1e-9)

        # Measurement update
        self.theta[a] = self.theta[a] + K * resid
        self.P[a] = self.P[a] - np.outer(K, context)
        # Symmetrize for numerical stability
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
# EVOLVE-BLOCK-END
