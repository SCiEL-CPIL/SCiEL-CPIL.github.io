"""Non-stationary linear contextual bandit policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Forgetting factor for non-stationarity
        self.forget = 0.99
        # Regularization
        self.reg = 1.0
        # Per-arm matrices A and vectors b
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Pre-compute Cholesky factors for efficiency
        self.L = [np.linalg.cholesky(np.eye(dim)) for _ in range(n_arms)]
        # Track time for deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # seconds, safety margin before 150s kill
        self._deadline_hit = False

    def _cholesky_update(self, arm, context):
        """Update A and Cholesky factor efficiently."""
        self.A[arm] = self.forget * self.A[arm] + np.outer(context, context)
        # Recompute Cholesky (dim is small, so this is fast)
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm] + 1e-6 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self.A[arm] = self.A[arm] + 0.1 * np.eye(self.d)
            self.L[arm] = np.linalg.cholesky(self.A[arm])

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            self._deadline_hit = True
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Compute theta = A^{-1} b
            L = self.L[a]
            # Solve L L^T theta = b  =>  theta = A^{-1} b
            # A = L L^T, so A^{-1} = L^{-T} L^{-1}
            # theta = L^{-T} (L^{-1} b)
            try:
                y = np.linalg.solve(L, self.b[a])
                theta = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)

            # Mean prediction
            mu = theta @ context

            # Variance: context^T A^{-1} context
            # A^{-1} x = L^{-T} L^{-1} x
            try:
                z = np.linalg.solve(L, context)
                var = z @ z
            except np.linalg.LinAlgError:
                var = 1.0

            # Thompson sampling: sample from N(mu, var)
            noise = np.random.randn()
            scores[a] = mu + noise * np.sqrt(max(var, 0.0))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        # Update with forgetting
        self.b[arm] = self.forget * self.b[arm] + reward * context
        self._cholesky_update(arm, context)
# EVOLVE-BLOCK-END
