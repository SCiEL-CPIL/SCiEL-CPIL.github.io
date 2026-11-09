"""Non-stationary linear contextual bandit with global model and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Linear contextual bandit using a single global model with arm-augmented features.
    Uses Thompson Sampling for exploration and a forgetting factor for non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented feature dimension: 1 (bias) + dim (context) + n_arms (one-hot arm)
        self.d_aug = 1 + dim + n_arms
        # Global model parameters
        self.A = np.eye(self.d_aug) * 1.0  # Regularized covariance
        self.b = np.zeros(self.d_aug)
        # Forgetting factor
        self.alpha = 0.99
        # Track time for deadline guard
        self._start_time = time.time()
        self._deadline = 145.0  # Safety margin before 150s limit
        self._round = 0

    def _make_feature(self, context, arm):
        """Create augmented feature vector: [1, context..., one_hot(arm)]."""
        feat = np.zeros(self.d_aug)
        feat[0] = 1.0  # bias
        feat[1:1 + self.d] = context  # context
        feat[1 + self.d + arm] = 1.0  # one-hot arm indicator
        return feat

    def select(self, context):
        """Select an arm using Thompson Sampling with the global model."""
        self._round += 1

        # Check deadline - if close to timeout, exploit best estimate
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest expected reward
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                feat = self._make_feature(context, a)
                theta = np.linalg.solve(self.A, self.b)
                score = feat @ theta
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Compute posterior mean and covariance
        theta = np.linalg.solve(self.A, self.b)
        # For Thompson Sampling, we need to sample from the posterior
        # Use Cholesky decomposition for efficiency
        try:
            L = np.linalg.cholesky(self.A)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.A + 1e-6 * np.eye(self.d_aug))

        # Sample theta_s ~ N(theta, A)
        # A = L @ L.T, so we need L^{-T} @ z where z ~ N(0, I)
        z = np.random.randn(self.d_aug)
        # Solve L.T @ v = z, then theta_s = theta + L @ v
        v = np.linalg.solve(L.T, z)
        theta_s = theta + L @ v

        # Score each arm using the sampled parameter
        scores = np.zeros(self.n)
        for a in range(self.n):
            feat = self._make_feature(context, a)
            scores[a] = feat @ theta_s

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global model with the observed reward."""
        feat = self._make_feature(context, arm)
        # Forgetting: shrink A and b, then add new observation
        self.A = self.alpha * self.A + np.outer(feat, feat)
        self.b = self.alpha * self.b + reward * feat
# EVOLVE-BLOCK-END
