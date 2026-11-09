"""Hierarchical Bayesian Linear Bandit with dual forgetting factors."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Hierarchical Bayesian Linear Bandit:
    - Global model: slow forgetting (tracks long-term trend)
    - Arm-specific model: fast forgetting (tracks immediate regime shifts)
    - Uses Thompson Sampling for exploration
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 110  # safety margin before 120s limit

        # Global (shared) model - slow forgetting
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)
        self.alpha_global = 0.999  # very slow forgetting

        # Arm-specific models - fast forgetting
        self.A_arm = [np.eye(dim) for _ in range(n_arms)]
        self.b_arm = [np.zeros(dim) for _ in range(n_arms)]
        self.alpha_arm = 0.98  # faster forgetting for regime shifts

        # Noise variance assumption
        self.sigma_sq = 1.0

        # Track elapsed time
        self.elapsed = 0.0

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        self.elapsed = time.time() - self.t0
        return self.elapsed > self.deadline

    def _sample_theta(self, A, b, context):
        """
        Sample a theta from the posterior N(theta_hat, sigma_sq * A^{-1}).
        Then compute theta @ context.
        """
        # Compute posterior mean and covariance
        # We need A^{-1} @ b for mean, and A^{-1} for covariance
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            inv_A = np.eye(self.d)

        theta_hat = inv_A @ b
        # Covariance of theta is sigma_sq * A^{-1}
        cov = self.sigma_sq * inv_A

        # Sample theta from N(theta_hat, cov)
        # Use Cholesky decomposition for sampling
        try:
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.d)
            theta_sample = theta_hat + L @ z
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            theta_sample = theta_hat + np.sqrt(np.maximum(np.diag(cov), 1e-10)) * np.random.randn(self.d)

        return theta_sample @ context

    def select(self, context):
        """Select arm using Thompson Sampling with hierarchical model."""
        # Check deadline
        if self._check_deadline():
            # Fallback: use greedy based on global model
            try:
                inv_A_global = np.linalg.inv(self.A_global)
                theta_global = inv_A_global @ self.b_global
            except np.linalg.LinAlgError:
                theta_global = self.b_global

            scores = []
            for a in range(self.n):
                try:
                    inv_A_arm = np.linalg.inv(self.A_arm[a])
                    theta_arm = inv_A_arm @ self.b_arm[a]
                except np.linalg.LinAlgError:
                    theta_arm = self.b_arm[a]

                # Combine global and arm-specific predictions
                pred = 0.5 * (theta_global @ context) + 0.5 * (theta_arm @ context)
                scores.append(pred)

            return int(np.argmax(scores))

        # Normal operation: Thompson Sampling
        scores = []
        for a in range(self.n):
            # Sample from global posterior
            theta_global_sample = self._sample_theta(self.A_global, self.b_global, context)

            # Sample from arm-specific posterior
            theta_arm_sample = self._sample_theta(self.A_arm[a], self.b_arm[a], context)

            # Combine predictions (weighted average)
            # Give more weight to arm-specific for recent regime shifts
            combined = 0.4 * theta_global_sample + 0.6 * theta_arm_sample
            scores.append(combined)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update both global and arm-specific models with new observation."""
        context = np.asarray(context)

        # Update global model (slow forgetting)
        self.A_global = self.alpha_global * self.A_global + np.outer(context, context)
        self.b_global = self.alpha_global * self.b_global + reward * context

        # Update arm-specific model (fast forgetting)
        self.A_arm[arm] = self.alpha_arm * self.A_arm[arm] + np.outer(context, context)
        self.b_arm[arm] = self.alpha_arm * self.b_arm[arm] + reward * context
# EVOLVE-BLOCK-END
