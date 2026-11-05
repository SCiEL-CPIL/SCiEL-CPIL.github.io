# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale adaptive Kalman filter for non-stationary linear bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # Adaptive process-noise tracking
        self.innov_hist = [[] for _ in range(n_arms)]
        self.innov_cap = 50
        self.base_Q = 1e-3          # baseline process noise
        self.adapt_gain = 1.0       # scaling for adaptive Q
        self.sigma2 = 1.0           # observation noise

        # Exploration
        self.alpha = 1.5
        self.pulled = [False] * n_arms

        # Deadline guard
        self._t0 = time.time()
        self._deadline = 140.0      # safety margin before 150s kill

    def _adaptive_Q(self, arm):
        """Compute adaptive process-noise covariance from innovation history."""
        hist = self.innov_hist[arm]
        if len(hist) >= 5:
            recent = np.array(hist[-self.innov_cap:])
            mean_sq = float(np.mean(recent ** 2))
            ratio = max(0.0, mean_sq / self.sigma2 - 1.0)
            Q = self.base_Q + self.adapt_gain * ratio
        else:
            Q = self.base_Q
        return Q

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            # Fallback: pick arm with highest mean estimate
            best = 0
            best_val = -np.inf
            for a in range(self.n):
                v = float(self.theta[a] @ context)
                if v > best_val:
                    best_val = v
                    best = a
            return int(best)

        x = np.asarray(context, dtype=float)
        scores = []
        for a in range(self.n):
            mean = float(self.theta[a] @ x)
            var = float(x @ self.P[a] @ x)
            if not self.pulled[a]:
                bonus = self.alpha * np.sqrt(var) + 2.0
            else:
                bonus = self.alpha * np.sqrt(var)
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=float)
        r = float(reward)
        a = arm

        # Prediction step (random walk)
        Q = self._adaptive_Q(a)
        self.P[a] = self.P[a] + Q * np.eye(self.d)

        # Innovation (prediction error)
        pred = float(self.theta[a] @ x)
        innov = r - pred
        self.innov_hist[a].append(innov)
        if len(self.innov_hist[a]) > self.innov_cap:
            self.innov_hist[a].pop(0)

        # Observation step (Kalman update, Joseph form)
        S = float(x @ self.P[a] @ x) + self.sigma2
        if S < 1e-12:
            K = np.zeros(self.d)
        else:
            K = (self.P[a] @ x) / S

        self.theta[a] = self.theta[a] + K * innov

        # Joseph form: P = (I - K x^T) P (I - K x^T)^T + K S K^T
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        self.P[a] = IKxT @ self.P[a] @ IKxT.T + np.outer(K, K) * S

        # Symmetrize for numerical stability
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)

        self.pulled[a] = True
# EVOLVE-BLOCK-END
