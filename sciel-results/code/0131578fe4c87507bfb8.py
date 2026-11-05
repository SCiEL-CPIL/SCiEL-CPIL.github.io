# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using per-arm Thompson Sampling
    with adaptive covariance inflation driven by standardized innovation.
    Replaces deterministic UCB with probabilistic arm selection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0

        # Posterior state per arm: theta (mean), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty for exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.q0 = 0.02
        self.q_max = 3.0
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.4

        # Thompson sampling parameters
        self.jitter = 1e-6
        self.step_count = 0
        self.warmup_steps = 40

        # For numerical stability in Cholesky
        self._chol_cache = [None] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov):
        """Compute adaptive process noise based on innovation EMA."""
        c = self.count[arm]
        if c < 3:
            return self.q0 * 3.0

        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - 1.5)
        q_scale = 1.0 + 2.0 * (1.0 - np.exp(-0.8 * excess))
        q_scale = min(q_scale, 50.0)

        q = min(self.q_max, self.q0 * q_scale)
        return q

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter
        self._chol_cache[a] = None  # invalidate cache

    def _update(self, a, ctx, reward):
        """Kalman update step: rank-1 update with innovation."""
        P = self.P[a]
        theta = self.theta[a]
        pred = float(theta @ ctx)
        innov = reward - pred
        var = float(ctx @ P @ ctx) + self.jitter
        K = P @ ctx / var
        self.theta[a] = theta + (K * innov)
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 0.0))
        self._chol_cache[a] = None
        self.count[a] += 1
        return innov

    def _sample_theta(self, a, ctx):
        """
        Sample a weight vector from the posterior N(theta, P).
        We only need the projected value on ctx, so we can be efficient:
        sample_theta = theta + L @ z where L L^T = P
        Then projected = ctx @ sample_theta = ctx @ theta + (L^T @ ctx) @ z
        """
        P = self.P[a]
        theta = self.theta[a]
        base = float(theta @ ctx)

        # Try Cholesky for sampling
        L = self._chol_cache[a]
        if L is None:
            try:
                L = np.linalg.cholesky(P)
                self._chol_cache[a] = L
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(P)
                    eigvals = np.maximum(eigvals, 0.0)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    self._chol_cache[a] = L
                except Exception:
                    return base  # no exploration if matrix is degenerate

        # Projected uncertainty: L^T @ ctx
        proj = L.T @ ctx
        # Sample from N(0, proj @ proj^T) -> just sample z ~ N(0, I) and take dot
        z = np.random.randn(self.d)
        sample_val = base + float(proj @ z)
        return sample_val

    def select(self, context):
        ctx = np.asarray(context, dtype=float)

        if self._check_deadline():
            # Greedy fallback
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1

        # Thompson sampling: sample one value per arm, pick the max
        samples = np.empty(self.n)
        for a in range(self.n):
            samples[a] = self._sample_theta(a, ctx)

        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)

        if self._check_deadline():
            return

        pred = float(self.theta[arm] @ ctx)
        innov = reward - pred

        q = self._adaptive_q(arm, innov)
        self._predict(arm, q)
        self._update(arm, ctx, reward)
# EVOLVE-BLOCK-END
