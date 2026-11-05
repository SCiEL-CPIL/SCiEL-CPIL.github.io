# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sliding-window linear bandit with per-arm recent-context covariance bonus."""

    def __init__(self, n_arms, dim, window=40, ridge=1e-3, beta0=1.0):
        self.n = n_arms
        self.d = dim
        self.window = window
        self.ridge = ridge
        self.beta0 = beta0
        # Per-arm ring buffers for contexts and rewards
        self.ctx_buf = [np.zeros((window, dim)) for _ in range(n_arms)]
        self.rw_buf = [np.zeros(window) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.ptr = [0 for _ in range(n_arms)]
        self.t = 0

    def _solve(self, arm):
        c = self.count[arm]
        if c < 2:
            # Not enough data: return zero estimate and large bonus direction
            return np.zeros(self.d), np.eye(self.d)
        X = self.ctx_buf[arm][:c]
        y = self.rw_buf[arm][:c]
        # Center reward by its mean to improve conditioning
        y_mean = y.mean()
        yc = y - y_mean
        A = X.T @ X + self.ridge * np.eye(self.d)
        b = X.T @ yc
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, b, rcond=None)[0]
        # Covariance of contexts in window for bonus scaling
        if c >= 2:
            Xc = X - X.mean(axis=0, keepdims=True)
            C = Xc.T @ Xc / (c - 1)
            C = C + self.ridge * np.eye(self.d)
        else:
            C = np.eye(self.d)
        return theta, C

    def select(self, context):
        context = np.asarray(context, dtype=float).ravel()
        scores = np.zeros(self.n)
        # Mild time-decay on exploration to settle later
        beta = self.beta0 * (1.0 / (1.0 + 0.002 * self.t))
        for a in range(self.n):
            theta, C = self._solve(a)
            mu = theta @ context
            # Bonus from covariance of recent contexts (inverse gives curvature)
            try:
                invC = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                invC = np.linalg.pinv(C)
            v = context @ invC @ context
            bonus = beta * np.sqrt(max(v, 0.0))
            scores[a] = mu + bonus
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float).ravel()
        self.ctx_buf[arm][self.ptr[arm]] = context
        self.rw_buf[arm][self.ptr[arm]] = reward
        self.ptr[arm] = (self.ptr[arm] + 1) % self.window
        if self.count[arm] < self.window:
            self.count[arm] += 1
        self.t += 1
# EVOLVE-BLOCK-END
