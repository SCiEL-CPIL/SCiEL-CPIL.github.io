"""Non-stationary linear contextual bandit: sliding-window per-arm ridge + Thompson
Sampling with global value model blend and a hard wall-clock guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Sliding-window per-arm ridge models with Thompson Sampling exploration,
    blended with a global value model, and a hard wall-clock guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 115.0          # safety margin vs. the 150 s kill
        self.elapsed = 0.0
        self.deadline_hit = False

        # Sliding window size for per-arm models (hard cutoff on old data)
        self.window_size = 100

        # Per-arm sliding-window data buffers
        self.arm_buffers = [[] for _ in range(n_arms)]

        # Per-arm ridge models (recomputed from window)
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) * 1.0 for _ in range(n_arms)]

        # Global linear value model (exponential forgetting, longer horizon)
        self.ridge = 1.0
        self.G = np.eye(dim) * self.ridge
        self.f = np.zeros(dim)
        self.theta = np.zeros(dim)
        self.theta_var = 1.0

        self.forgetting = 0.98
        self.alpha = 1.0
        self.last_pred = 0.0
        self.last_reward = 0.0
        self.count = 0

    def _check_deadline(self):
        if self.deadline_hit:
            return
        now = time.time()
        self.elapsed = now - self.t0
        if self.elapsed >= self.deadline:
            self.deadline_hit = True

    def _recompute_arm(self, a):
        """Recompute ridge model for arm a from its sliding window."""
        buf = self.arm_buffers[a]
        if len(buf) == 0:
            self.A[a] = np.eye(self.d) * 1.0
            self.b[a] = np.zeros(self.d)
            self.A_inv[a] = np.eye(self.d) * 1.0
            return
        # Stack contexts
        X = np.array([c for c, r in buf])
        R = np.array([r for c, r in buf])
        self.A[a] = X.T @ X + np.eye(self.d) * 1.0
        self.b[a] = X.T @ R
        try:
            self.A_inv[a] = np.linalg.inv(self.A[a])
        except np.linalg.LinAlgError:
            self.A_inv[a] = np.eye(self.d) * 1.0

    def _update_global(self):
        inv = np.linalg.inv(self.G + self.ridge * np.eye(self.d))
        self.theta = inv @ self.f
        self.theta_var = 0.5 * np.mean(np.abs(np.diag(inv)))

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)

        if self.deadline_hit:
            return int(np.argmax(self.theta @ ctx))

        self._update_global()

        scores = np.empty(self.n)
        for a in range(self.n):
            inv = self.A_inv[a]
            theta_a = inv @ self.b[a]
            # Thompson Sampling: sample from posterior
            v = inv @ ctx
            mean = theta_a @ ctx
            var = ctx @ v
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            scores[a] = mean + self.alpha * np.random.normal(0.0, std)

        # Blend with global value model
        global_score = self.theta @ ctx
        base = 0.5 * scores + 0.5 * global_score
        return int(np.argmax(base))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self.deadline_hit:
            return

        # --- Sliding window update for per-arm model ---
        self.arm_buffers[arm].append((ctx.copy(), reward))
        if len(self.arm_buffers[arm]) > self.window_size:
            self.arm_buffers[arm].pop(0)
        self._recompute_arm(arm)

        # --- Global value model update (exponential forgetting) ---
        f = self.forgetting
        self.G = f * self.G + np.outer(ctx, ctx)
        self.f = f * self.f + reward * ctx
        self._update_global()

        self.last_pred = self.theta @ ctx
        self.count += 1
# EVOLVE-BLOCK-END
