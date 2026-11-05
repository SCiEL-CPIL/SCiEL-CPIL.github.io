# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive non-stationary linear bandit with error-driven forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.t0 = time.time()

        # Per-arm linear model state
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting parameters
        self.base_gamma = 0.95
        self.gamma_min = 0.78
        self.gamma = [self.base_gamma for _ in range(n_arms)]
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.pred = [0.0 for _ in range(n_arms)]

        # Exploration
        self.C = 1.2
        self.explore_decay = 1000.0

        # Track best arm for deadline safety
        self.best_arm = 0
        self.best_score = -1e9

    def _elapsed(self):
        return time.time() - self.t0

    def _compute_gamma(self, arm, error):
        """Adaptively reduce gamma when prediction error is high (regime change)."""
        self.error_ema[arm] = 0.9 * self.error_ema[arm] + 0.1 * error
        # Stronger error-driven reduction
        reduction = min(1.0, self.error_ema[arm] * 0.1)
        self.gamma[arm] = max(self.gamma_min, self.base_gamma - reduction)
        # Slowly recover toward base_gamma
        self.gamma[arm] = 0.98 * self.gamma[arm] + 0.02 * self.base_gamma
        return self.gamma[arm]

    def select(self, context):
        # Deadline guard: if close to limit, return best known arm
        if self._elapsed() > 140.0:
            return self.best_arm

        self.t += 1
        ctx = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
                mu = theta @ ctx
                # Exploration bonus with slow decay
                bonus = self.C * np.sqrt(max(0.0, ctx @ inv_A @ ctx))
                bonus *= (1.0 / (1.0 + self.t / self.explore_decay))
                scores[a] = mu + bonus
            except np.linalg.LinAlgError:
                scores[a] = 0.0

        arm = int(np.argmax(scores))
        # Track best for deadline safety
        if scores[arm] > self.best_score:
            self.best_score = scores[arm]
            self.best_arm = arm

        return arm

    def update(self, context, arm, reward):
        if self._elapsed() > 140.0:
            return

        ctx = np.asarray(context, dtype=np.float64)
        r = float(reward)
        a = int(arm)

        # Compute prediction error for adaptive gamma
        try:
            inv_A = np.linalg.inv(self.A[a])
            theta = inv_A @ self.b[a]
            pred = theta @ ctx
        except np.linalg.LinAlgError:
            pred = 0.0

        self.pred[a] = pred
        error = (r - pred) ** 2
        gamma = self._compute_gamma(a, error)

        # Update with adaptive forgetting
        self.A[a] = gamma * self.A[a] + np.outer(ctx, ctx)
        self.b[a] = gamma * self.b[a] + r * ctx

        # Periodic ridge to prevent numerical issues
        if self.t % 100 == 0:
            for i in range(self.n):
                self.A[i] += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
