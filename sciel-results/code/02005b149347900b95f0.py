# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-scale ensemble linear bandit with model selection via residual variance.
    Each arm maintains a bank of regressors with geometrically spaced forgetting factors.
    The policy selects the arm whose ensemble member with the lowest recent residual
    variance gives the highest predicted reward.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.n_arms = n_arms

        # Geometrically spaced forgetting factors (memory horizons)
        # Short memory (fast adaptation) to long memory (stability)
        self.decays = [0.90, 0.93, 0.96, 0.98, 0.995]
        self.n_scales = len(self.decays)

        # Regularization
        self.ridge = 1.0

        # Per-arm, per-scale state
        # A_inv[arm][scale] : (dim, dim)
        # b[arm][scale]     : (dim,)
        # residual_sq[arm][scale] : scalar, EMA of squared residuals
        self.A_inv = [[[np.eye(dim) * self.ridge for _ in range(self.n_scales)] for _ in range(n_arms)]]
        self.b = [[[np.zeros(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]]
        self.residual_ema = [[[1.0 for _ in range(self.n_scales)] for _ in range(n_arms)]]

        # Pre-compute inverses
        self.A_inv = [[[np.linalg.inv(self.A_inv[a][s]) for s in range(self.n_scales)] for a in range(n_arms)]]

        # Tracking
        self.t = 0
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.recent_rewards = []

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

        # Exploration
        self.base_beta = 1.0
        self.beta = self.base_beta

    def _check_deadline(self):
        if time.time() > self.deadline:
            return True
        return False

    def _update_adaptive_params(self):
        """Update beta based on observed reward statistics."""
        if len(self.recent_rewards) > 200:
            self.recent_rewards = self.recent_rewards[-200:]

        if len(self.recent_rewards) >= 10:
            rewards = np.array(self.recent_rewards)
            self.reward_mean = float(np.mean(rewards))
            self.reward_var = max(float(np.var(rewards)), 1e-6)
        else:
            self.reward_mean = 0.0
            self.reward_var = 1.0

        # Adapt beta: more exploration early, scaled by variance
        time_factor = 1.0 / (1.0 + self.t * 0.005)
        var_factor = 1.0 + np.sqrt(self.reward_var) * 0.3
        self.beta = self.base_beta * var_factor * (0.5 + 0.5 * time_factor)

    def _smw_update(self, arm, scale, x):
        """Sherman-Morrison-Woodbury update for A_inv after A <- decay*A + xx^T."""
        decay = self.decays[scale]
        A_inv_old = self.A_inv[arm][scale]

        # A_new = decay * A_old + x x^T
        # A'^{-1} = (1/decay) * A_old^{-1}
        A_prime_inv = A_inv_old / decay

        Ax = A_prime_inv @ x
        denom = 1.0 + x @ Ax

        self.A_inv[arm][scale] = A_prime_inv - np.outer(Ax, Ax) / denom

    def select(self, context):
        x = np.asarray(context, dtype=np.float64).flatten()

        if self._check_deadline():
            # Greedy fallback
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                # Use the best scale (lowest residual) for greedy
                best_s = int(np.argmin(self.residual_ema[a]))
                theta = self.A_inv[a][best_s] @ self.b[a][best_s]
                score = theta @ x
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # For each arm, find the best scale (lowest residual variance)
        # and compute UCB score using that scale's model
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Find scale with lowest residual EMA
            best_s = int(np.argmin(self.residual_ema[a]))
            A_inv = self.A_inv[a][best_s]
            theta = A_inv @ self.b[a][best_s]
            mean = theta @ x
            variance = x @ A_inv @ x
            scores[a] = mean + self.beta * np.sqrt(max(variance, 0.0))

        # Small random tie-breaker
        scores += np.random.normal(0, 1e-6, size=self.n)

        best_arm = int(np.argmax(scores))
        self.t += 1
        return best_arm

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).flatten()
        r = float(reward)

        # Store reward
        self.recent_rewards.append(r)

        # Update all scales for this arm
        for s in range(self.n_scales):
            # Predict before updating (for residual)
            A_inv = self.A_inv[arm][s]
            theta = A_inv @ self.b[arm][s]
            pred = theta @ x

            # Residual
            resid = r - pred
            resid_sq = resid * resid

            # Update residual EMA
            decay_r = 0.95
            self.residual_ema[arm][s] = decay_r * self.residual_ema[arm][s] + (1 - decay_r) * resid_sq

            # Update model
            self._smw_update(arm, s, x)
            self.b[arm][s] = self.decays[s] * self.b[arm][s] + r * x

        # Adaptively update beta
        self._update_adaptive_params()
# EVOLVE-BLOCK-END
