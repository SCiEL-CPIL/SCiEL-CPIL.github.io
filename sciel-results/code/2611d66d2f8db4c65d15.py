# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Exponentially-weighted linear bandit with Sherman-Morrison rank-1 updates and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as small ridge for faster initial learning
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Precompute inverses for Sherman-Morrison updates
        self.A_inv = [np.eye(dim) / 0.1 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting factor - start moderate for fast initial adaptation
        self.gamma = 0.95
        # Exploration parameter - higher for better initial exploration
        self.alpha = 1.5
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s
        self.round_count = 0
        # Track per-arm visit counts for adaptive exploration
        self.visit_counts = [0] * n_arms
        # Track prediction errors for adaptive gamma
        self.error_history = []

    def _sherman_morrison_update(self, arm, context, reward):
        """Update A_inv and theta using Sherman-Morrison formula after rank-1 update."""
        x = context
        # A_new = gamma * A + x x^T
        # We maintain A_inv such that A_inv @ A = I
        # After update: A_new = gamma * A + outer(x, x)
        # A_new_inv = (gamma * A + x x^T)^{-1}
        # Using Sherman-Morrison: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_new = gamma * A + x x^T = gamma (A + (1/gamma) x x^T)
        # So A_new^{-1} = (1/gamma) (A + (1/gamma) x x^T)^{-1}
        # = (1/gamma) [A^{-1} - A^{-1} (x/gamma) (x/gamma)^T A^{-1} / (1 + (x/gamma)^T A^{-1} (x/gamma))]
        
        inv = self.A_inv[arm]
        x_over_g = x / self.gamma
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse
            self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        # A_new^{-1} = (1/gamma) * (inv - np.outer(Ax, Ax) / denom)
        new_inv = (inv - np.outer(Ax, Ax) / denom) / self.gamma
        self.A_inv[arm] = new_inv
        
        # Update b: b_new = gamma * b + reward * x
        self.b[arm] = self.gamma * self.b[arm] + reward * x
        # Update theta = A_new^{-1} @ b_new
        self.theta[arm] = new_inv @ self.b[arm]
        
        # Also update A for potential fallback use
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return a deterministic arm based on context hash to avoid exceptions
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.A_inv[a]
            mean = theta @ context
            # UCB bonus with adaptive exploration
            var = context @ inv @ context
            # Use per-arm visit count to adjust exploration
            # Arms with fewer visits get slightly higher bonus
            visit_bonus = 0.1 * np.log1p(1.0 / (1.0 + self.visit_counts[a]))
            bonus = self.alpha * np.sqrt(max(var, 0.0)) + visit_bonus
            scores.append(mean + bonus)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.visit_counts[arm] += 1
        
        # Adaptive forgetting: increase gamma (slower forgetting) when prediction is good,
        # decrease gamma (faster forgetting) when prediction error is large
        pred = self.theta[arm] @ context
        pred_err = abs(reward - pred)
        
        # Store error for trend detection
        self.error_history.append(pred_err)
        if len(self.error_history) > 50:
            self.error_history.pop(0)
        
        # Compute recent error trend
        if len(self.error_history) >= 10:
            recent_err = np.mean(self.error_history[-10:])
            older_err = np.mean(self.error_history[:-10]) if len(self.error_history) > 10 else recent_err
            err_trend = recent_err - older_err
        else:
            err_trend = 0.0
        
        # Target gamma: lower when error is high or increasing (faster adaptation)
        # Higher when error is low and stable (slower forgetting to accumulate info)
        target_gamma = 0.999 - 0.099 * np.tanh(pred_err / 1.5)
        # Additional adjustment for trend: if errors are increasing, forget faster
        if err_trend > 0.1:
            target_gamma -= 0.02
        elif err_trend < -0.1:
            target_gamma += 0.01
        
        self.gamma = 0.90 * self.gamma + 0.10 * target_gamma
        self.gamma = np.clip(self.gamma, 0.88, 0.999)
        
        # Perform the update
        self._sherman_morrison_update(arm, context, reward)
        
        # Adjust exploration: decay alpha over time but keep some exploration
        # Faster decay early, slower later
        decay_rate = 0.9998 if self.round_count > 100 else 0.9995
        self.alpha = max(0.5, self.alpha * decay_rate)
# EVOLVE-BLOCK-END
