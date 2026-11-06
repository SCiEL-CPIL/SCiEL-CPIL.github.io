"""Non-stationary linear contextual bandit with global model and arm identity features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear model with arm identity as explicit features.
    Uses exponential forgetting to handle non-stationarity.
    The feature vector is [context, one_hot_arm] for prediction,
    but since we predict per-arm, we use a single global model
    where the arm is part of the feature space.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Global model: features are [context (d), arm_one_hot (n)] -> total dim d + n
        self.d_full = dim + n_arms
        # Regularized inverse covariance
        self.P = np.eye(self.d_full) * 1.0
        # Weight vector
        self.theta = np.zeros(self.d_full)
        # Forgetting factor
        self.beta = 0.99
        # Exploration parameter
        self.alpha = 1.0
        # Track last time each arm was played for adaptive exploration
        self.arm_last_played = [-1] * n_arms
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s

    def _augment(self, context, arm):
        """Create augmented feature vector [context, one_hot_arm]."""
        feat = np.zeros(self.d_full)
        feat[:self.d] = context
        feat[self.d + arm] = 1.0
        return feat

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return best arm based on current theta
            scores = []
            for a in range(self.n):
                feat = self._augment(context, a)
                scores.append(feat @ self.theta)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            feat = self._augment(context, a)
            # Predicted reward
            pred = feat @ self.theta
            # Uncertainty
            v = self.P @ feat
            ucb = pred + self.alpha * np.sqrt(feat @ v)
            # Adaptive exploration: boost arms not played recently
            time_since_played = self.t - self.arm_last_played[a]
            if time_since_played > 0:
                ucb += 0.1 * np.log(1 + time_since_played)
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        self.arm_last_played[arm] = self.t

        feat = self._augment(context, arm)
        # Forgetting: shrink P and theta
        self.P *= self.beta
        self.theta *= self.beta

        # RLS update
        P_feat = self.P @ feat
        denom = 1.0 + feat @ P_feat
        gain = P_feat / denom
        self.P -= np.outer(gain, P_feat)
        pred = feat @ self.theta
        error = reward - pred
        self.theta += gain * error
# EVOLVE-BLOCK-END
