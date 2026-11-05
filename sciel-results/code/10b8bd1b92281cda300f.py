"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 135.0  # tighter safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        # Start with ridge for numerical stability
        ridge = 0.2
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.93
        self.gamma_min = 0.70
        self.gamma_max = 0.99
        self.C = 1.2  # exploration coefficient
        self.C_min = 0.3
        self.C_max = 5.0

        # For tracking prediction errors to adapt forgetting
        self.pred_errors = [0.0] * n_arms
        self.error_counts = [0] * n_arms

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on prediction error."""
        c = self.error_counts[arm]
        if c < 8:
            return self.base_gamma

        # Use recent prediction error to adjust gamma
        # Higher prediction error -> faster forgetting (lower gamma)
        mean_err = self.pred_errors[arm] / c
        # Scale error to adjust gamma
        err_scale = min(mean_err, 5.0)
        gamma = self.base_gamma - 0.10 * err_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 2.0
        # Decay exploration over time
        time_decay = max(0.4, 1.0 - 0.15 * (self.t / 500.0))
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(max(0.0, context @ self.V[a] @ context))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        self.pred_errors[arm] += err * err
        self.error_counts[arm] += 1

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d) * (1.0 + 0.2)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic numerical stabilization: clip eigenvalues
        if self.t % 50 == 0:
            for a in range(self.n):
                V = self.V[a]
                # Ensure symmetry
                V = (V + V.T) / 2.0
                # Clip to ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(V)
                eigvals = np.clip(eigvals, 1e-8, 100.0)
                self.V[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
