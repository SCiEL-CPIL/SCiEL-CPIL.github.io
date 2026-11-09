"""Linear Thompson Sampling with adaptive forgetting for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with adaptive forgetting factor."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._inv_A = [np.eye(dim) for _ in range(n_arms)]
        self._theta = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting: start moderate, adjust based on prediction error
        self.gamma = 0.98
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        # Track prediction errors for adaptation
        self._recent_errors = []
        self._max_error_history = 50
        # Noise variance assumption
        self.sigma_sq = 1.0
        # Time tracking
        self._start_time = time.time()
        self._deadline = 145.0  # Safety margin before 150s kill

    def _update_inverse(self, arm):
        """Recompute inverse and theta for an arm after update."""
        try:
            self._inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Regularize if singular
            self.A[arm] += 1e-6 * np.eye(self.d)
            self._inv_A[arm] = np.linalg.inv(self.A[arm])
        self._theta[arm] = self._inv_A[arm] @ self.b[arm]

    def _adapt_forgetting(self, context, arm, reward):
        """Adapt forgetting factor based on prediction error."""
        pred = self._theta[arm] @ context
        err = abs(reward - pred)
        self._recent_errors.append(err)
        if len(self._recent_errors) > self._max_error_history:
            self._recent_errors.pop(0)
        
        if len(self._recent_errors) >= 10:
            mean_err = np.mean(self._recent_errors[-10:])
            # High error -> increase forgetting (lower gamma) to adapt faster
            # Low error -> decrease forgetting (higher gamma) to exploit
            if mean_err > 2.0:
                self.gamma = max(self.gamma_min, self.gamma * 0.95)
            elif mean_err < 0.5:
                self.gamma = min(self.gamma_max, self.gamma * 1.02)
            else:
                # Gentle adjustment toward 0.98
                self.gamma += (0.98 - self.gamma) * 0.05

    def select(self, context):
        """Select arm via Thompson Sampling."""
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fall back to greedy with current theta estimates
            scores = [self._theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        samples = []
        for a in range(self.n):
            # Sample theta from N(theta_hat, inv_A)
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(self._inv_A[a] + 1e-8 * np.eye(self.d))
                z = np.random.randn(self.d)
                theta_sample = self._theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback to diagonal approximation
                var_diag = np.diag(self._inv_A[a])
                theta_sample = self._theta[a] + np.random.randn(self.d) * np.sqrt(np.maximum(var_diag, 1e-8))
            samples.append(theta_sample @ context)
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """Update posterior with forgetting."""
        context = np.asarray(context, dtype=np.float64)
        g = self.gamma
        self.A[arm] = g * self.A[arm] + np.outer(context, context)
        self.b[arm] = g * self.b[arm] + reward * context
        self._update_inverse(arm)
        self._adapt_forgetting(context, arm, reward)
# EVOLVE-BLOCK-END
