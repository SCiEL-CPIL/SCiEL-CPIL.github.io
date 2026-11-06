"""Adaptive linear bandit policy for non-stationary settings."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Per-arm linear bandit with adaptive forgetting.
    
    Model: reward = x^T theta_a + noise
    - Each arm has its own linear parameter vector theta_a.
    - Adaptive forgetting factor reacts to prediction errors.
    - Adaptive UCB alpha based on recent error magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm linear models
        # A[a]: dim x dim matrix
        # b[a]: dim vector
        # inv_A[a]: dim x dim matrix (inverse of A[a])
        self.A = np.array([np.eye(dim) for _ in range(n_arms)])
        self.b = np.array([np.zeros(dim) for _ in range(n_arms)])
        self.inv_A = np.array([np.eye(dim) for _ in range(n_arms)])

        # Adaptive forgetting
        self.base_lambda = 0.97
        self.min_lambda = 0.75
        self.max_lambda = 0.998
        self.error_buffer = deque(maxlen=25)

        # Adaptive UCB
        self.base_alpha = 1.0
        self.min_alpha = 0.5
        self.max_alpha = 2.5

        # Reward scale tracking for normalized errors
        self.reward_magnitude = 1.0
        self._reward_count = 0

        # Track predictions for error computation
        self._last_mu = np.zeros(n_arms)

        # Time guard
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0

        # Round counter
        self._round = 0

    def _compute_lambda(self):
        """Adaptive forgetting factor based on recent prediction errors."""
        if len(self.error_buffer) < 5:
            return self.base_lambda

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Normalize by reward magnitude for scale-invariant adaptation
        norm_scale = max(self.reward_magnitude * 0.2, 0.05)
        normalized = (mean_err + std_err * 0.5) / norm_scale

        if normalized < 0.3:
            return self.max_lambda
        elif normalized > 2.0:
            return self.min_lambda
        else:
            t = (normalized - 0.3) / (2.0 - 0.3)
            t_smooth = t * t * (3 - 2 * t)
            return self.max_lambda - t_smooth * (self.max_lambda - self.min_lambda)

    def _compute_alpha(self):
        """Adaptive exploration parameter."""
        if len(self.error_buffer) < 8:
            return self.base_alpha

        recent = list(self.error_buffer)
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0

        # Scale-invariant normalization
        norm_scale = max(self.reward_magnitude * 0.2, 0.05)
        normalized = (mean_err + std_err * 0.5) / norm_scale

        if normalized < 0.4:
            return self.min_alpha
        elif normalized > 2.5:
            return self.max_alpha
        else:
            t = (normalized - 0.4) / (2.5 - 0.4)
            t_smooth = t * t * (3 - 2 * t)
            return self.min_alpha + t_smooth * (self.max_alpha - self.min_alpha)

    def select(self, context):
        # Deadline guard
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        self._round += 1

        # Get current adaptive parameters
        alpha = self._compute_alpha()

        scores = np.zeros(self.n)
        mus = np.zeros(self.n)
        for a in range(self.n):
            # Compute theta_a
            theta_a = self.inv_A[a] @ self.b[a]
            
            # Prediction
            mu = theta_a @ context
            mus[a] = mu
            
            # Variance: x^T inv_A[a] x
            var = context @ (self.inv_A[a] @ context)
            std = np.sqrt(max(0.0, var))

            # UCB with relative confidence
            scores[a] = mu + alpha * std
            self._last_mu[a] = mu

        # Soft-max selection for stability: pick argmax but with slight randomness
        # when scores are close
        best_score = np.max(scores)
        close_arms = np.where(np.abs(scores - best_score) < 0.01 * max(1.0, np.abs(best_score)))[0]
        if len(close_arms) > 1 and self._round > 10:
            return int(close_arms[np.random.randint(len(close_arms))])
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self._deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Update reward magnitude tracking (exponential moving average)
        abs_reward = abs(reward)
        self._reward_count += 1
        alpha_mag = 0.05 if self._reward_count < 100 else 0.02
        self.reward_magnitude = (1 - alpha_mag) * self.reward_magnitude + alpha_mag * abs_reward

        # Compute prediction error BEFORE updating
        pred = self._last_mu[arm]
        error = abs(reward - pred)
        self.error_buffer.append(error)

        # Adaptive forgetting factor
        lam = self._compute_lambda()

        # Update per-arm linear model with forgetting
        # A_new = lam * A_old + x x^T
        # b_new = lam * b_old + r * x
        A_old = self.A[arm]
        b_old = self.b[arm]
        
        A_new = lam * A_old + np.outer(context, context)
        b_new = lam * b_old + reward * context

        # Update inverse via Sherman-Morrison: (lam*A + xx^T)^{-1}
        # = (1/lam) * (A^{-1} - A^{-1} x x^T A^{-1} / (lam + x^T A^{-1} x))
        inv_A_old = self.inv_A[arm]
        Ax = inv_A_old @ context
        xAx = context @ Ax
        denom = lam + xAx
        
        if abs(denom) > 1e-12:
            inv_A_new = (1.0 / lam) * (inv_A_old - np.outer(Ax, Ax) / denom)
        else:
            # Fallback: recompute
            try:
                inv_A_new = np.linalg.inv(A_new)
            except np.linalg.LinAlgError:
                inv_A_new = np.eye(self.d)

        # Store updates
        self.A[arm] = A_new
        self.b[arm] = b_new
        self.inv_A[arm] = inv_A_new

# EVOLVE-BLOCK-END
