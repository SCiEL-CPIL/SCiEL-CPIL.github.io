"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive LinUCB with Sherman-Morrison inverse-covariance updates,
    adaptive forgetting factor, and deadline guard for non-stationary
    linear contextual bandits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = self._t0 + 115.0  # safety margin before 120s kill

        # Store V = (A)^{-1} for efficient Sherman-Morrison updates
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.97
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.base_C = 1.5
        self.C_min = 0.5
        self.C_max = 3.0

        # Track recent rewards for adaptive tuning
        self.reward_history = []
        self.max_history = 100

        # Per-arm recent reward tracking for adaptive forgetting
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_reward_counts = [0] * n_arms

        # Error tracking
        self.error_ema = 0.0
        self.error_alpha = 0.20

    def _check_deadline(self):
        """Return True if we should stop exploring and just pick best known arm."""
        return time.time() > self._deadline

    def _adaptive_gamma(self, arm, context, reward):
        """Adapt forgetting factor based on recent reward variance and prediction error."""
        if len(self.reward_history) < 10:
            return self.base_gamma

        recent = self.reward_history[-min(50, len(self.reward_history)):]
        var = np.var(recent)
        mean_abs = np.mean(np.abs(recent))

        # Higher variance in rewards suggests more non-stationarity -> lower gamma
        # Normalize variance relative to mean absolute reward
        if mean_abs > 1e-9:
            normalized_var = var / (mean_abs ** 2 + 1e-9)
        else:
            normalized_var = 0.0

        # Also compute prediction error
        inv = self.V[arm]
        theta = inv @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * err

        # Combine signals: high variance OR high error -> faster forgetting
        norm_var = min(normalized_var, 1.0)
        norm_err = min(err / (1.0 + self.error_ema), 1.0)
        nonstationarity = 0.6 * norm_var + 0.4 * norm_err

        # Map to gamma: high non-stationarity -> lower gamma
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * nonstationarity
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration parameter based on arm count and overall progress."""
        total_count = sum(self.counts)
        if total_count < 5:
            return self.C_max

        # Decrease exploration as we learn more
        progress = min(1.0, total_count / 500.0)
        C = self.C_max - (self.C_max - self.C_min) * progress

        # Increase exploration for arms with fewer samples
        if self.counts[arm] < total_count / self.n:
            C *= 1.2

        return max(self.C_min, min(self.C_max, C))

    def select(self, context):
        if self._check_deadline():
            # Deadline approaching: pick the arm with highest estimated reward
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # Uncertainty bonus
            v = self.V[a]
            uncertainty = np.sqrt(context @ v @ context)
            C = self._adaptive_C(a)
            scores.append(mu + C * uncertainty)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        gamma = self._adaptive_gamma(arm, context, reward)

        # Sherman-Morrison update with forgetting factor
        # A_new = gamma * A_old + x x^T
        # V_new = (A_new)^{-1} = (gamma * A_old + x x^T)^{-1}
        # Using Sherman-Morrison on the scaled matrix:
        # V_new = (1/gamma) * V_old - (1/gamma)^2 * (V_old @ x @ x^T @ V_old) / (gamma + x^T @ V_old @ x)

        V = self.V[arm]
        x = context
        gamma_inv = 1.0 / gamma

        Vx = V @ x
        xVx = x @ Vx  # scalar

        denom = 1.0 + gamma_inv * xVx

        # V_new = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom
        self.V[arm] = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom

        # Symmetrize to avoid numerical drift
        self.V[arm] = 0.5 * (self.V[arm] + self.V[arm].T)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * x

        self.counts[arm] += 1

        # Track reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        # Track per-arm reward stats
        self.arm_reward_sums[arm] = gamma * self.arm_reward_sums[arm] + reward
        self.arm_reward_counts[arm] = gamma * self.arm_reward_counts[arm] + 1
# EVOLVE-BLOCK-END
