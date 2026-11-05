"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Exponentially weighted linear regression (EWLR) with adaptive forgetting
    - Sherman-Morrison-like O(d^2) updates for inverse covariance
    - UCB exploration with adaptive beta
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        
        # Per-arm state: A_inv (d x d), b (d,), and reward tracking
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # For adaptive forgetting: track per-arm prediction error EMA
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.pred_ema = [0.0 for _ in range(n_arms)]
        self.error_ema = [1.0 for _ in range(n_arms)]
        
        # Adaptive forgetting factor
        self.base_gamma = 0.95
        self.min_gamma = 0.85
        self.max_gamma = 0.99
        
        # Exploration parameter
        self.beta0 = 1.0
        self.counts = [0 for _ in range(n_arms)]
        
        # Deadline safety
        self.deadline = 145.0  # seconds from start, leave 5s margin

    def _get_gamma(self, arm):
        """Adaptive forgetting factor based on prediction error."""
        # Higher error -> lower gamma (faster forgetting)
        err = self.error_ema[arm]
        # Map error to gamma in [min_gamma, max_gamma]
        # If error is high, forget faster
        gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * min(err / 2.0, 1.0)
        return max(self.min_gamma, min(self.max_gamma, gamma))

    def select(self, context):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Return a safe default arm
            return 0
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            if self.counts[a] == 0:
                # No data yet: pure exploration
                scores[a] = self.beta0 * np.sqrt(context @ self.A_inv[a] @ context)
            else:
                theta = self.A_inv[a] @ self.b[a]
                mu = theta @ context
                var = context @ self.A_inv[a] @ context
                beta = self.beta0 * np.sqrt(1.0 + self.t * 0.001)
                scores[a] = mu + beta * np.sqrt(max(var, 1e-8))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.counts[arm] += 1
        
        # Update reward and prediction EMAs for adaptive forgetting
        theta = self.A_inv[arm] @ self.b[arm]
        pred = theta @ context
        self.reward_ema[arm] = 0.9 * self.reward_ema[arm] + 0.1 * reward
        self.pred_ema[arm] = 0.9 * self.pred_ema[arm] + 0.1 * pred
        self.error_ema[arm] = 0.9 * self.error_ema[arm] + 0.1 * (reward - pred) ** 2
        
        # Adaptive forgetting factor
        gamma = self._get_gamma(arm)
        
        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Update A_inv using Sherman-Morrison-like approach with forgetting
        # A_new = gamma * A_old + x x^T
        # We want A_inv_new = (gamma * A_old + x x^T)^{-1}
        # Using matrix inversion lemma: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_old_inv = self.A_inv[arm], so A_old = A_old_inv^{-1}
        # But we have A_old_inv, and we want (gamma * A_old + x x^T)^{-1}
        # = (gamma * (A_old_inv^{-1}) + x x^T)^{-1}
        # Let B = gamma * A_old_inv^{-1}, then (B + x x^T)^{-1} = B^{-1} - B^{-1} x x^T B^{-1} / (1 + x^T B^{-1} x)
        # B^{-1} = (1/gamma) * A_old_inv
        # So: A_inv_new = (1/gamma)*A_old_inv - [(1/gamma)*A_old_inv * x * x^T * (1/gamma)*A_old_inv] / (1 + x^T (1/gamma)*A_old_inv x)
        # = (1/gamma)*A_old_inv - [(1/gamma^2)*A_old_inv x x^T A_old_inv] / (1 + (1/gamma)*x^T A_old_inv x)
        
        A_inv_old = self.A_inv[arm]
        Ax = A_inv_old @ context
        xA = context @ A_inv_old  # same as Ax^T if A_inv is symmetric
        denom = 1.0 + (1.0 / gamma) * (context @ Ax)
        
        # A_inv_new = (1/gamma) * A_inv_old - (1/gamma^2) * Ax x^T A_inv_old / denom
        # = (1/gamma) * A_inv_old - (1/gamma^2) * np.outer(Ax, context) @ A_inv_old / denom
        # Actually: Ax x^T A_inv_old = outer(Ax, x @ A_inv_old) = outer(Ax, Ax) since A_inv is symmetric
        # Wait: x^T A_inv_old is a row vector, Ax = A_inv_old x is a column vector
        # outer(Ax, context @ A_inv_old) = outer(Ax, Ax) if A_inv is symmetric
        # Let's be careful: A_inv_old is symmetric, so context @ A_inv_old = (A_inv_old @ context)^T = Ax^T
        # So outer(Ax, context @ A_inv_old) = outer(Ax, Ax^T) = np.outer(Ax, Ax)
        
        A_inv_new = (1.0 / gamma) * A_inv_old - (1.0 / (gamma * gamma)) * np.outer(Ax, Ax) / denom
        
        # Symmetrize to reduce numerical drift
        A_inv_new = 0.5 * (A_inv_new + A_inv_new.T)
        
        self.A_inv[arm] = A_inv_new
# EVOLVE-BLOCK-END
