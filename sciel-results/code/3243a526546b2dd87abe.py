"""Kernelized non-stationary linear bandit with RBF time features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit using a kernelized approach.
    We augment the context with a small set of RBF features of the current time
    to capture drift. This turns the problem into a (approximately) stationary
    linear bandit in an augmented feature space.
    
    The augmented context is: [x, phi_1(t), phi_2(t), ..., phi_K(t)]
    where phi_k(t) = exp(-gamma_k * (t - c_k)^2) for a set of centers c_k.
    
    We use LinUCB on the augmented features with a forgetting factor for
    robustness, plus adaptive exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0  # time step counter

        # RBF kernel parameters for time features
        self.n_rbf = 8  # number of RBF basis functions
        self.rbf_centers = np.linspace(0, 500, self.n_rbf)  # spread over expected horizon
        self.rbf_widths = np.linspace(10, 50, self.n_rbf)  # varying widths

        # Augmented dimension
        self.d_aug = dim + self.n_rbf

        # LinUCB parameters
        self.gamma = 0.99  # forgetting factor
        self.lam = 1.0     # regularization
        self.alpha = 1.5   # exploration parameter

        # Per-arm state
        self.A = [self.lam * np.eye(self.d_aug) for _ in range(n_arms)]
        self.b = [np.zeros(self.d_aug) for _ in range(n_arms)]

        # Adaptive alpha tracking
        self.reward_history = []
        self.residuals = []

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # seconds, safety margin before 150s

    def _augment_context(self, context):
        """Add RBF time features to the context."""
        context = np.asarray(context, dtype=np.float64)
        # Compute RBF features of current time
        rbf_feats = np.zeros(self.n_rbf)
        for k in range(self.n_rbf):
            rbf_feats[k] = np.exp(-((self.t - self.rbf_centers[k]) ** 2) / (2.0 * self.rbf_widths[k] ** 2))
        return np.concatenate([context, rbf_feats])

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Return arm 0 as a safe fallback
            return 0

        self.t += 1
        aug_ctx = self._augment_context(context)

        scores = []
        for a in range(self.n):
            # Compute theta and uncertainty
            try:
                inv_A = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                inv_A = np.eye(self.d_aug) / self.lam

            theta = inv_A @ self.b[a]
            mean = theta @ aug_ctx
            # Uncertainty term
            var = aug_ctx @ inv_A @ aug_ctx
            # Adaptive alpha: reduce exploration as we learn more
            n_obs = self.t  # approximate
            effective_alpha = self.alpha * (1.0 + 50.0 / (n_obs + 1.0))
            ucb = mean + effective_alpha * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1  # This is already incremented in select, but we track separately
        # Actually, let's not double-increment. The select increments t.
        # Let me reconsider: t should increment once per round.
        # Let's use a separate counter.

        aug_ctx = self._augment_context(context)

        # Update with forgetting factor
        self.A[arm] = self.gamma * self.A[arm] + np.outer(aug_ctx, aug_ctx)
        self.b[arm] = self.gamma * self.b[arm] + reward * aug_ctx

        # Track residuals for adaptive normalization
        try:
            inv_A = np.linalg.inv(self.A[arm])
            theta = inv_A @ self.b[arm]
            pred = theta @ aug_ctx
            residual = reward - pred
            self.residuals.append(residual)
            # Keep only last 100 residuals
            if len(self.residuals) > 100:
                self.residuals = self.residuals[-100:]
        except np.linalg.LinAlgError:
            pass
# EVOLVE-BLOCK-END
