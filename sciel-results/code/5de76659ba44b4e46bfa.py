"""Non-stationary linear contextual bandit: Thompson Sampling with sliding window
per-arm linear models for fast adaptation to non-stationary drift."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with sliding-window per-arm ridge models.
    Each arm maintains a circular buffer of recent (context, reward) pairs.
    On select, sample theta from the posterior and pick argmax of sampled scores."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 115.0
        self.elapsed = 0.0
        self.deadline_hit = False

        # Sliding window size: number of recent observations per arm
        self.window_size = 50
        self.ridge = 1.0

        # Per-arm circular buffers
        self.buf_ctx = [np.zeros((self.window_size, dim)) for _ in range(n_arms)]
        self.buf_rwd = [np.zeros(self.window_size) for _ in range(n_arms)]
        self.buf_ptr = [0] * n_arms
        self.buf_count = [0] * n_arms

        # Precomputed per-arm model parameters (theta_a, A_inv)
        self.theta_a = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) * 1.0 for _ in range(n_arms)]

        self.count = 0

    def _check_deadline(self):
        if self.deadline_hit:
            return
        now = time.time()
        self.elapsed = now - self.t0
        if self.elapsed >= self.deadline:
            self.deadline_hit = True

    def _recompute_arm(self, a):
        """Recompute ridge solution for arm a from its sliding window."""
        cnt = self.buf_count[a]
        if cnt == 0:
            self.theta_a[a] = np.zeros(self.d)
            self.A_inv[a] = np.eye(self.d) * 1.0
            return
        X = self.buf_ctx[a][:cnt]
        y = self.buf_rwd[a][:cnt]
        A = X.T @ X + self.ridge * np.eye(self.d)
        b = X.T @ y
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d) * 1.0
        self.A_inv[a] = A_inv
        self.theta_a[a] = A_inv @ b

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)

        if self.deadline_hit:
            # Exploit: use current point estimates
            scores = np.array([self.theta_a[a] @ ctx for a in range(self.n)])
            return int(np.argmax(scores))

        # Thompson Sampling: sample theta from posterior for each arm
        sampled_scores = np.empty(self.n)
        for a in range(self.n):
            theta_mean = self.theta_a[a]
            cov = self.A_inv[a]
            # Sample theta ~ N(theta_mean, cov)
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(self.d))
            z = np.random.randn(self.d)
            theta_sample = theta_mean + L @ z
            sampled_scores[a] = theta_sample @ ctx

        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self.deadline_hit:
            return

        # Add to circular buffer
        ptr = self.buf_ptr[arm]
        cnt = self.buf_count[arm]
        self.buf_ctx[arm][ptr] = ctx
        self.buf_rwd[arm][ptr] = reward
        self.buf_ptr[arm] = (ptr + 1) % self.window_size
        if cnt < self.window_size:
            self.buf_count[arm] = cnt + 1

        # Recompute this arm's model
        self._recompute_arm(arm)
        self.count += 1
# EVOLVE-BLOCK-END
