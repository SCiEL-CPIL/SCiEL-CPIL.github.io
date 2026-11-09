"""Non-stationary linear contextual bandit with a single global model and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear model with arm-one-hot features and adaptive exponential forgetting.
    Uses LinUCB-style exploration with a single parameter vector over the augmented feature space.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: dim context + n_arms one-hot
        self.d_aug = dim + n_arms

        # Initialize the global model
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        self.A_inv = np.eye(self.d_aug)

        # Forgetting factor (adaptive)
        self.forgetting = 0.995
        self.base_forgetting = 0.995
        self.min_forgetting = 0.95
        self.max_forgetting = 0.999

        # Error tracking for adaptive forgetting
        self.recent_errors = []
        self.error_window = 50

        # UCB exploration parameter
        self.alpha = 1.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s

    def _augment(self, context, arm):
        """Create augmented feature vector: [context, one-hot(arm)]"""
        feat = np.zeros(self.d_aug)
        feat[:self.d] = context
        feat[self.d + arm] = 1.0
        return feat

    def _update_forgetting(self, reward_error):
        """Adaptively adjust the forgetting factor based on recent prediction errors."""
        self.recent_errors.append(reward_error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        if len(self.recent_errors) >= 10:
            mean_err = np.mean(np.abs(self.recent_errors))
            # Higher error -> lower forgetting (more reactive)
            # Lower error -> higher forgetting (more stable)
            # Map mean_err from [0, ~5] to forgetting from [max, min]
            norm_err = min(mean_err / 3.0, 1.0)
            self.forgetting = self.max_forgetting - norm_err * (self.max_forgetting - self.min_forgetting)
            self.forgetting = np.clip(self.forgetting, self.min_forgetting, self.max_forgetting)

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Return a safe default arm
            return 0

        # Compute scores for all arms using the global model
        theta = self.A_inv @ self.b
        scores = np.zeros(self.n)
        for a in range(self.n):
            feat = self._augment(context, a)
            pred = theta @ feat
            # Uncertainty: sqrt(feat^T A_inv feat)
            var = feat @ self.A_inv @ feat
            scores[a] = pred + self.alpha * np.sqrt(max(var, 0.0))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        feat = self._augment(context, arm)

        # Compute prediction error for adaptive forgetting
        theta = self.A_inv @ self.b
        pred = theta @ feat
        error = reward - pred
        self._update_forgetting(error)

        # Update with forgetting
        f = self.forgetting
        self.A = f * self.A + np.outer(feat, feat)
        self.b = f * self.b + reward * feat

        # Update inverse using Sherman-Morrison or direct inversion for small dims
        # For efficiency with small d_aug, use direct inversion
        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            # Fallback: regularize
            self.A += 1e-6 * np.eye(self.d_aug)
            self.A_inv = np.linalg.inv(self.A)
# EVOLVE-BLOCK-END
