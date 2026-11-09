"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Hybrid UCB/Thompson Sampling with drift detection for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0  # safety margin before 120s kill

        # Per-arm state
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.last_reward = [0.0 for _ in range(n_arms)]
        self.last_context = [np.zeros(dim) for _ in range(n_arms)]
        self.last_t = [-1 for _ in range(n_arms)]

        # Forgetting factor
        self.alpha = 0.99

        # Exploration parameters
        self.ucb_beta = 0.5
        self.ts_sigma = 1.0

        # Drift detection
        self.drift_threshold = 2.0
        self.drift_window = 5

        # Pre-allocate for speed
        self._rng = np.random.default_rng()

    def _reset_arm(self, a):
        """Reset a single arm's state to initial."""
        self.A[a] = np.eye(self.d)
        self.b[a] = np.zeros(self.d)
        self.last_reward[a] = 0.0
        self.last_context[a] = np.zeros(self.d)
        self.last_t[a] = -1

    def _check_drift(self, a, context, reward):
        """Check if arm 'a' has experienced drift since last pull."""
        if self.last_t[a] < 0:
            return False
        # Simple drift check: if reward deviates significantly from predicted
        # We can check if the reward is far from what we'd expect
        # For simplicity, use a sliding window approach
        # If we haven't seen this arm in a while, reset
        if self.t - self.last_t[a] > self.drift_window:
            return True
        return False

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return a default arm to avoid timeout
            return 0

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Compute theta_a = A_a^{-1} b_a
            # Use Cholesky for stability and speed
            try:
                L = np.linalg.cholesky(self.A[a])
                # Solve L y = b, then L^T theta = y
                y = np.linalg.solve(L, self.b[a])
                theta = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                theta = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]

            # Compute variance: context^T A^{-1} context
            try:
                z = np.linalg.solve(L, context)
                var = np.dot(z, z)
            except np.linalg.LinAlgError:
                var = np.dot(context, np.linalg.lstsq(self.A[a], context, rcond=None)[0])

            mean = np.dot(theta, context)

            # Hybrid score: UCB + Thompson Sampling noise
            # UCB component
            ucb_score = mean + self.ucb_beta * np.sqrt(var)

            # Thompson Sampling component: sample from posterior
            # theta ~ N(theta_hat, var * A^{-1})
            # We approximate by adding Gaussian noise scaled by sqrt(var)
            noise = self._rng.standard_normal() * self.ts_sigma * np.sqrt(var)

            # Combine: use UCB as base, add TS noise for stochastic exploration
            score = ucb_score + noise
            scores.append(score)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Check for drift before updating
        if self._check_drift(arm, context, reward):
            self._reset_arm(arm)

        # Update with forgetting factor
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context

        # Store for drift detection
        self.last_reward[arm] = reward
        self.last_context[arm] = context.copy()
        self.last_t[arm] = self.t

        self.t += 1
# EVOLVE-BLOCK-END
