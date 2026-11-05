"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling for non-stationary linear bandits with adaptive covariance inflation."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm: A (precision matrix), b (cross-product vector)
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Forgetting factor base
        self.forgetting = 0.98
        # Adaptive inflation parameters
        self.base_inflation = 1.0
        self.inflation_alpha = 0.1
        # Track per-arm reward statistics for innovation detection
        self.prev_rewards = [0.0] * n_arms
        self.reward_std = [1.0] * n_arms
        self.reward_mean = [0.0] * n_arms
        self.counts = [0] * n_arms
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit
        self._rng = np.random.default_rng(42)

    def _check_deadline(self):
        return time.time() - self.start_time > self.deadline

    def select(self, context):
        # Deadline guard: if we're running out of time, pick argmax of mean predictions
        if self._check_deadline():
            scores = [self.b[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Compute mean and variance for this arm
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.eye(self.d) * 0.01

            theta_mean = A_inv @ self.b[a]
            mean_score = theta_mean @ context

            # Variance of the predicted reward
            var = context @ A_inv @ context
            std = np.sqrt(max(var, 1e-10))

            # Thompson sampling: sample from posterior
            theta_sample = theta_mean + self._rng.normal(0, 1, self.d) * std / np.sqrt(self.d + 1)
            # More principled: sample from N(theta_mean, A_inv)
            # But for efficiency, use the scalar variance approximation
            sample_score = mean_score + self._rng.normal(0, std)
            scores.append(sample_score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update reward statistics for adaptive inflation
        self.counts[arm] += 1
        n = self.counts[arm]
        # Exponential moving average for mean and std
        alpha_s = 0.1
        self.reward_mean[arm] = (1 - alpha_s) * self.reward_mean[arm] + alpha_s * reward
        diff = reward - self.prev_rewards[arm] if n > 1 else 0.0
        self.reward_std[arm] = (1 - alpha_s) * self.reward_std[arm] + alpha_s * abs(diff)
        self.prev_rewards[arm] = reward

        # Adaptive inflation: increase uncertainty when innovation is large
        innovation = abs(reward - self.reward_mean[arm])
        std_ref = max(self.reward_std[arm], 0.1)
        z = innovation / std_ref if std_ref > 1e-6 else 0.0
        inflation = self.base_inflation + self.inflation_alpha * min(z, 3.0)

        # Apply forgetting and inflation
        decay = self.forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context

        # Inflate the covariance (i.e., shrink precision) to reflect non-stationarity
        # This is done by scaling A down (which increases A_inv)
        # But we need to be careful: we want to increase uncertainty
        # A better approach: add inflation to the diagonal of A^{-1}
        # Since we store A (precision), we can scale it down
        self.A[arm] /= inflation
# EVOLVE-BLOCK-END
