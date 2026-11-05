"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with time-varying forgetting factor and exploration tuning."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor: start aggressive, increase over time
        self.beta_min = 0.90
        self.beta_max = 0.995
        self.beta_ramp = 500  # number of steps to ramp beta

        # Exploration parameter
        self.alpha = 1.5

        # Track time steps for adaptive alpha
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit

        # Track reward statistics for potential adaptation
        self.reward_sum = [0.0 for _ in range(n_arms)]
        self.reward_sq_sum = [0.0 for _ in range(n_arms)]
        self.reward_count = [0 for _ in range(n_arms)]

    def _get_beta(self):
        """Get current forgetting factor based on time."""
        if self.t < self.beta_ramp:
            # Linear ramp from beta_min to beta_max
            progress = self.t / self.beta_ramp
            beta = self.beta_min + progress * (self.beta_max - self.beta_min)
        else:
            beta = self.beta_max
        return beta

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            return 0

        scores = []
        for a in range(self.n):
            # Use Cholesky decomposition for numerical stability and speed
            try:
                L = np.linalg.cholesky(self.A[a] + 1e-8 * np.eye(self.d))
                # Solve L * v = context
                v = np.linalg.solve(L, context)
                # Solve L.T * w = v
                w = np.linalg.solve(L.T, v)
                inv_context = w @ context  # context.T @ A^{-1} @ context
            except np.linalg.LinAlgError:
                # Fallback to inverse if Cholesky fails
                inv = np.linalg.inv(self.A[a] + 1e-8 * np.eye(self.d))
                inv_context = context @ inv @ context

            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(self.A[a] + 1e-8 * np.eye(self.d), self.b[a])
            except np.linalg.LinAlgError:
                inv = np.linalg.inv(self.A[a] + 1e-8 * np.eye(self.d))
                theta = inv @ self.b[a]

            mean = theta @ context
            std = np.sqrt(max(0, inv_context))

            # Adaptive alpha: decrease over time to reduce exploration
            # Use a slower decay to maintain more exploration
            if self.t > 0:
                alpha_t = self.alpha / (1.0 + 0.1 * np.log1p(self.t))
            else:
                alpha_t = self.alpha

            scores.append(mean + alpha_t * std)

        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track reward statistics
        self.reward_count[arm] += 1
        self.reward_sum[arm] += reward
        self.reward_sq_sum[arm] += reward * reward

        # Get current forgetting factor
        beta_t = self._get_beta()

        # Update with adaptive forgetting factor
        self.A[arm] = beta_t * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta_t * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
