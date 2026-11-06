"""Non-stationary linear contextual bandit with adaptive forgetting and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit with:
    - Per-arm ridge regression with adaptive forgetting factor
    - UCB exploration with adaptive beta
    - Sliding window for stability
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 110  # Safety margin before 120s limit

        # Per-arm state
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / 0.1 for _ in range(n_arms)]

        # Adaptive forgetting factor
        self.base_forgetting = 0.98
        self.forgetting = self.base_forgetting

        # Exploration parameter
        self.beta = 1.0
        self.beta_min = 0.5
        self.beta_max = 3.0

        # Track recent rewards for adaptive exploration
        self.reward_history = []
        self.max_history_len = 50

        # Per-arm reward tracking for adaptive forgetting
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_counts = [0] * n_arms
        self.arm_prev_reward = [0.0] * n_arms

    def select(self, context):
        self.t += 1

        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mean = theta @ context
            variance = context @ self.A_inv[a] @ context
            scores[a] = mean + self.beta * np.sqrt(variance)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1

        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history_len:
            self.reward_history.pop(0)

        # Adaptive forgetting: increase forgetting when rewards are volatile
        if len(self.reward_history) >= 10:
            recent = np.array(self.reward_history[-20:])
            vol = np.std(recent)
            # Higher volatility -> more aggressive forgetting
            self.forgetting = np.clip(self.base_forgetting - 0.02 * vol, 0.9, 0.999)

        # Update per-arm state with forgetting
        f = self.forgetting
        self.A[arm] = f * self.A[arm] + np.outer(context, context)
        self.b[arm] = f * self.b[arm] + reward * context

        # Update inverse using Sherman-Morrison-Woodbury
        # (A + xx^T)^{-1} = A^{-1} - A^{-1}xx^T A^{-1} / (1 + x^T A^{-1} x)
        # But we have f*A + xx^T, which is trickier.
        # For simplicity, recompute inverse periodically or use Woodbury on the full update.
        # Let's use: (fA + xx^T)^{-1} where we treat this as update from previous A_inv
        # Actually, let's just recompute inverse every few steps for numerical stability
        if self.t % 5 == 0 or self.t < 10:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d) * 10.0
        else:
            # Approximate update: scale existing inverse
            # (fA + xx^T)^{-1} ≈ (1/f) * (A + (1/f)xx^T)^{-1}
            # This is approximate. Let's use Woodbury properly.
            # We want (fA + xx^T)^{-1}
            # = (f(A + (1/f)xx^T))^{-1}
            # = (1/f) * (A + (1/f)xx^T)^{-1}
            # Using Woodbury: (A + uv^T)^{-1} = A^{-1} - A^{-1}uv^T A^{-1}/(1+v^T A^{-1} u)
            # Here u = x, v = x/f
            A_inv = self.A_inv[arm]
            u = context
            v = context / f
            denom = 1.0 + v @ A_inv @ u
            if abs(denom) > 1e-10:
                update = np.outer(A_inv @ u, A_inv @ v) / denom
                self.A_inv[arm] = A_inv - update
            else:
                try:
                    self.A_inv[arm] = np.linalg.inv(self.A[arm])
                except np.linalg.LinAlgError:
                    self.A_inv[arm] = np.eye(self.d) * 10.0

        # Adaptive beta: increase exploration when uncertainty is high or rewards are volatile
        if len(self.reward_history) >= 5:
            recent_std = np.std(self.reward_history[-10:])
            # Base beta on volatility
            self.beta = np.clip(1.0 + 0.5 * recent_std, self.beta_min, self.beta_max)

        # Track per-arm statistics
        self.arm_reward_sums[arm] = f * self.arm_reward_sums[arm] + reward
        self.arm_counts[arm] = f * self.arm_counts[arm] + 1
        self.arm_prev_reward[arm] = reward

    def _get_lambda(self):
        """Return current regularization/forgetting parameter."""
        return 1.0 - self.forgetting
# EVOLVE-BLOCK-END
