# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with variance-driven forgetting and efficient solves."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.n_arms = n_arms

        # Base decay factor; we will adapt it slightly.
        self.base_decay = 0.99
        self.decay = self.base_decay

        # Regularization
        self.ridge = 1.0

        # Per-arm state
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Pre-compute inverse; use Sherman-Morrison for rank-1 updates
        self.A_inv = [np.linalg.inv(self.A[i]) for i in range(n_arms)]
        self.dirty = [False] * n_arms

        # Tracking for adaptive exploration
        self.recent_rewards = []
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.t = 0

        # Per-arm reward tracking for better adaptation
        self.arm_rewards = [[] for _ in range(n_arms)]

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115  # 5s safety margin out of 120s

        # Exploration scale
        self.base_beta = 1.0
        self.beta = self.base_beta

    def _check_deadline(self):
        """Return True if we are close to the deadline."""
        if time.time() > self.deadline:
            return True
        return False

    def _update_adaptive_params(self):
        """Update decay and beta based on observed reward statistics."""
        # Keep a rolling window of rewards
        if len(self.recent_rewards) > 200:
            self.recent_rewards = self.recent_rewards[-200:]

        if len(self.recent_rewards) >= 10:
            rewards = np.array(self.recent_rewards)
            self.reward_mean = np.mean(rewards)
            self.reward_var = max(np.var(rewards), 1e-6)
        else:
            self.reward_mean = 0.0
            self.reward_var = 1.0

        # Adapt decay: if variance is high, we might want to adapt faster (lower decay)
        # If variance is low, we can be more stable (higher decay)
        # Use a smoother mapping with a narrower range for stability
        var_scale = min(self.reward_var / 3.0, 1.0)  # cap influence
        self.decay = 0.998 - 0.03 * var_scale  # range [0.968, 0.998]

        # Adapt beta: increase exploration if variance is high or early on
        # Use a slower decay for time_factor
        time_factor = 1.0 / (1.0 + self.t * 0.005)  # decays more slowly
        var_factor = 1.0 + np.sqrt(self.reward_var) * 0.3
        self.beta = self.base_beta * var_factor * (0.5 + 0.5 * time_factor)

    def _smw_update(self, arm, x):
        """Sherman-Morrison-Woodbury update for A_inv after A <- decay*A + xx^T."""
        # A_new = decay * A_old + x x^T
        # A_inv_new = (decay * A_old + x x^T)^{-1}
        # = (1/decay) * (A_old + (1/decay) x x^T)^{-1}
        # = (1/decay) * (A_inv_old - (1/decay) A_inv_old x x^T A_inv_old / (decay + x^T A_inv_old x))
        # Actually, let's use the standard SMW formula:
        # If B = A + u v^T, then B^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_new = decay * A_old + x x^T
        # Let A' = decay * A_old, so A_new = A' + x x^T
        # A'^{-1} = (1/decay) * A_old^{-1}
        # A_new^{-1} = A'^{-1} - A'^{-1} x x^T A'^{-1} / (1 + x^T A'^{-1} x)
        
        A_inv_old = self.A_inv[arm]
        decay = self.decay
        
        # A'^{-1} = (1/decay) * A_old^{-1}
        A_prime_inv = A_inv_old / decay
        
        # x^T A'^{-1} x
        Ax = A_prime_inv @ x
        denom = 1.0 + x @ Ax
        
        # A_new^{-1} = A'^{-1} - A'^{-1} x x^T A'^{-1} / denom
        # = A'^{-1} - (Ax * x^T A'^{-1}) / denom
        # = A'^{-1} - (Ax @ (x^T A_prime_inv)) / denom
        # x^T A_prime_inv = (A_prime_inv @ x)^T = Ax^T (since symmetric)
        
        self.A_inv[arm] = A_prime_inv - np.outer(Ax, Ax) / denom
        
        # Update A for consistency (used in case we need to recompute)
        self.A[arm] = decay * self.A[arm] + np.outer(x, x)

    def select(self, context):
        # Convert context to numpy array for consistent operations
        x = np.asarray(context, dtype=np.float64).flatten()

        # Check deadline; if close, use a deterministic greedy choice based on current estimates
        if self._check_deadline():
            # Greedy fallback: pick arm with highest estimated mean
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                score = theta @ x
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Normal selection with UCB
        scores = np.zeros(self.n)
        for a in range(self.n):
            A_inv = self.A_inv[a]
            theta = A_inv @ self.b[a]
            mean = theta @ x
            variance = x @ A_inv @ x
            # Use adaptive beta
            scores[a] = mean + self.beta * np.sqrt(max(variance, 0.0))

        # Add small random tie-breaker to avoid systematic bias
        scores += np.random.normal(0, 1e-6, size=self.n)

        best_arm = int(np.argmax(scores))
        self.t += 1
        return best_arm

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).flatten()
        r = float(reward)

        # Store reward for adaptive parameter tuning
        self.recent_rewards.append(r)
        self.arm_rewards[arm].append(r)
        if len(self.arm_rewards[arm]) > 50:
            self.arm_rewards[arm] = self.arm_rewards[arm][-50:]

        # Use Sherman-Morrison-Woodbury for efficient inverse update
        self._smw_update(arm, x)
        self.b[arm] = self.decay * self.b[arm] + r * x

        # Adaptively update decay and beta
        self._update_adaptive_params()
# EVOLVE-BLOCK-END
