"""Non-stationary linear contextual bandit: Thompson Sampling with sliding window."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Thompson Sampling with a sliding-window ridge-regression posterior.

    Rewards are modeled as r = x^T w_a + noise, where the arm weight w_a
    drifts over time. We keep a fixed-length window of recent (x, r) pairs
    per arm and fit a ridge-regression posterior on that window each step.
    Exploration is done via Thompson Sampling (posterior sampling).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 80          # fixed window for non-stationarity
        self.ridge = 1.0               # regularization on the window Gram matrix
        self.noise = 1.0               # assumed observation noise variance

        # Per-arm storage of recent (context, reward) pairs.
        self.histories = [deque(maxlen=self.window_size) for _ in range(self.n)]

        # Cached per-arm posterior (recomputed in _refresh).
        self.theta = [np.zeros(dim) for _ in range(self.n)]
        self.Ainv = [np.eye(dim) for _ in range(self.n)]
        self.cov = [self.noise * np.eye(dim) for _ in range(self.n)]

        # Deadline guard.
        self.t0 = time.time()
        self.deadline = 120.0          # hard task limit; stay well under it
        self.safety_margin = 5.0       # seconds to leave before the limit
        self._rng = np.random.default_rng(0)

    def _refresh(self, a):
        """Recompute the ridge-regression posterior for arm a from its window."""
        d = self.d
        if len(self.histories[a]) == 0:
            self.theta[a] = np.zeros(d)
            self.Ainv[a] = np.eye(d)
            self.cov[a] = self.noise * np.eye(d)
            return
        X = np.stack([x for (x, r) in self.histories[a]])
        y = np.array([r for (x, r) in self.histories[a]])
        A = X.T @ X + self.ridge * np.eye(d)
        b = X.T @ y
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        self.theta[a] = Ainv @ b
        self.Ainv[a] = Ainv
        self.cov[a] = self.noise * Ainv

    def select(self, context):
        context = np.asarray(context, dtype=float).ravel()

        # Deadline guard: near the limit, fall back to greedy (no sampling cost
        # beyond what we already have) to guarantee we return in time.
        if (time.time() - self.t0) > (self.deadline - self.safety_margin):
            return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))

        scores = []
        for a in range(self.n):
            self._refresh(a)
            # Posterior sample: theta ~ N(theta_hat, cov)
            theta_sample = self.theta[a] + self._rng.multivariate_normal(
                np.zeros(self.d), self.cov[a]
            )
            scores.append(float(theta_sample @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float).ravel()
        self.histories[arm].append((context, float(reward)))
        self._refresh(arm)
# EVOLVE-BLOCK-END
