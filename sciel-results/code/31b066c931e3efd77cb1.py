"""Non-stationary linear contextual bandit: adaptive-forgetting Thompson Sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Thompson Sampling with adaptive forgetting for a non-stationary linear bandit.

    Each arm keeps a posterior over its weight vector w (reward ~ x^T w + noise).
    A per-arm rolling prediction-error drives an adaptive forgetting factor gamma:
    high error (regime change) -> small gamma (forget fast, explore more);
    low error (stationary)     -> large gamma (retain information).
    An explicit wall-clock deadline guard degrades to greedy exploitation near the limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Posterior parameters: p(w | data) = N(theta, (lam*I + sum xx^T)^-1 * sigma^2)
        self.lam = 1.0            # ridge / prior strength
        self.sigma2 = 1.0         # assumed reward noise variance
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting state
        self.gamma = 0.995        # base forgetting factor
        self.gamma_min = 0.85     # aggressive forgetting on detected change
        self.gamma_max = 0.999    # near-stationary retention
        self.err = [0.0 for _ in range(n_arms)]   # rolling prediction error (EMA)
        self.err_base = 1.0       # baseline error scale for normalization
        self.err_alpha = 0.1      # EMA smoothing for the error statistic
        self.err_window = 0.15    # how much the error can move gamma
        # Deadline guard
        self.t0 = time.time()
        self.deadline = 150.0 - 3.0   # safety margin before the 150s kill
        self.deadline_hit = False

    def _time_ok(self):
        if self.deadline_hit:
            return False
        if time.time() >= self.deadline:
            self.deadline_hit = True
        return not self.deadline_hit

    def _posterior_mean_cov(self, a):
        inv = np.linalg.inv(self.A[a])
        theta = inv @ self.b[a]
        cov = inv * self.sigma2
        return theta, cov

    def select(self, context):
        x = np.asarray(context, dtype=float)
        # Deadline guard: exploit only, no sampling overhead, deterministic.
        if not self._time_ok():
            best_a, best_val = 0, -np.inf
            for a in range(self.n):
                theta, _ = self._posterior_mean_cov(a)
                v = float(theta @ x)
                if v > best_val:
                    best_val, best_a = v, a
            return int(best_a)

        # Thompson sampling: draw one weight vector per arm, pick argmax.
        vals = np.empty(self.n)
        for a in range(self.n):
            theta, cov = self._posterior_mean_cov(a)
            # Sample w ~ N(theta, cov). Use Cholesky for a stable draw.
            try:
                L = np.linalg.cholesky(cov)
                w = theta + L @ np.random.standard_normal(self.d)
            except np.linalg.LinAlgError:
                # Fallback: diagonal jitter and retry once, else mean.
                try:
                    L = np.linalg.cholesky(cov + 1e-6 * np.eye(self.d))
                    w = theta + L @ np.random.standard_normal(self.d)
                except np.linalg.LinAlgError:
                    w = theta
            vals[a] = float(w @ x)
        return int(np.argmax(vals))

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=float)
        a = int(arm)
        r = float(reward)

        # 1) Adaptive forgetting: track per-arm prediction error (EMA), then set gamma.
        #    Predicted mean reward under current posterior.
        theta, _ = self._posterior_mean_cov(a)
        pred = float(theta @ x)
        err_now = (r - pred) ** 2
        self.err[a] = (1.0 - self.err_alpha) * self.err[a] + self.err_alpha * err_now
        # Normalize error against a baseline; larger error -> smaller gamma.
        scale = self.err_base + self.err[a]
        t = self.err[a] / scale                 # in [0, ~1]
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * t
        gamma = float(np.clip(gamma, self.gamma_min, self.gamma_max))
        self.gamma = gamma

        # 2) Apply forgetting to the sufficient statistics, then add the new observation.
        #    Forgetting the ridge term keeps A well-conditioned (never fully shrinks to 0).
        self.A[a] = gamma * (self.A[a] - self.lam * np.eye(self.d)) + self.lam * np.eye(self.d)
        self.A[a] = self.A[a] + np.outer(x, x)
        self.b[a] = gamma * self.b[a] + r * x
# EVOLVE-BLOCK-END
