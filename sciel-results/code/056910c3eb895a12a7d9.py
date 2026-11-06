"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with non-stationary handling and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 115.0  # 5s safety margin before 120s
        self._recent_errors = []
        self._base_lambda = 0.96
        self._min_lambda = 0.85
        self._max_lambda = 0.995
        self._error_window = 20
        self._last_arm = None
        self._last_context = None
        self._last_reward = None
        self._round = 0

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 5:
            return self._base_lambda
        recent = self._recent_errors[-self._error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0
        # High error or high variance -> forget faster (lower lambda)
        # Low error and variance -> retain more (higher lambda)
        # Map mean_err to lambda adjustment
        # If mean error is large relative to std, we're in a regime change
        ratio = mean_err / (std_err + 1e-8)
        # ratio > 2 suggests systematic drift, reduce lambda
        # ratio < 0.5 suggests stable, increase lambda
        if ratio > 2.0:
            lambda_val = self._min_lambda
        elif ratio > 1.0:
            lambda_val = self._min_lambda + (self._base_lambda - self._min_lambda) * (1.0 - (ratio - 1.0))
        elif ratio < 0.5:
            lambda_val = self._base_lambda + (self._max_lambda - self._base_lambda) * (0.5 - ratio)
        else:
            lambda_val = self._base_lambda
        return np.clip(lambda_val, self._min_lambda, self._max_lambda)

    def _predict(self, arm, context):
        """Predict reward for given arm and context."""
        try:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mu = theta @ context
            # Uncertainty: context^T A^{-1} context
            v = context @ A_inv @ context
            return mu, np.sqrt(max(v, 0.0))
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(self.A[arm]) @ self.b[arm]
            mu = theta @ context
            v = context @ np.linalg.pinv(self.A[arm]) @ context
            return mu, np.sqrt(max(v, 0.0))

    def select(self, context):
        self._round += 1
        # Deadline guard: if we're close to the limit, just pick the arm with highest mean
        if time.time() > self._deadline - 2.0:
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.inv(self.A[a]) @ self.b[a]
                    scores.append(theta @ context)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu, sigma = self._predict(a, context)
            # Adaptive UCB: use a moderate alpha
            alpha = 0.7
            scores.append(mu + alpha * sigma)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track prediction error for adaptive forgetting
        if self._last_arm is not None and self._last_arm == arm and self._last_context is not None:
            # We predicted this arm previously; compute error
            try:
                theta = np.linalg.inv(self.A[arm]) @ self.b[arm]
                pred = theta @ self._last_context
                err = abs(reward - pred)
                self._recent_errors.append(err)
                if len(self._recent_errors) > self._error_window * 2:
                    self._recent_errors = self._recent_errors[-self._error_window:]
            except np.linalg.LinAlgError:
                pass

        self._last_arm = arm
        self._last_context = context.copy() if hasattr(context, 'copy') else np.array(context)
        self._last_reward = reward

        # Adaptive forgetting factor
        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
