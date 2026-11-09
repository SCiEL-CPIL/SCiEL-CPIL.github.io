"""Non-stationary linear contextual bandit with adaptive per-arm forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy with adaptive per-arm forgetting.
    
    Uses LinUCB-style estimates with per-arm adaptive forgetting factors that
    increase when recent prediction errors suggest the arm's weights have changed.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting: base rate and per-arm current rate
        self.base_forget = 0.99
        self.forget_rates = [self.base_forget] * n_arms
        
        # Track recent prediction errors per arm for non-stationarity detection
        self.recent_errors = [[] for _ in range(n_arms)]
        self.error_window = 10
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        self._update_theta()

    def _update_theta(self):
        """Update theta estimates from current A and b."""
        for a in range(self.n):
            self.theta[a] = self.inv_A[a] @ self.b[a]

    def _detect_regime_change(self, arm, pred_error):
        """Check if recent errors suggest a regime change for this arm."""
        errors = self.recent_errors[arm]
        errors.append(pred_error)
        if len(errors) > self.error_window:
            errors.pop(0)
        
        if len(errors) < 3:
            return False
        
        # If mean absolute error is high relative to std, weights may have changed
        mean_abs = np.mean(np.abs(errors))
        # Simple heuristic: if recent errors are consistently large, adapt faster
        return mean_abs > 0.5

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            pred = theta @ context
            # UCB bonus
            var = context @ self.inv_A[a] @ context
            bonus = self.alpha * np.sqrt(max(var, 0.0))
            scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction error for non-stationarity detection
        pred = self.theta[arm] @ context
        pred_error = reward - pred
        
        # Detect regime change and adjust forgetting rate
        if self._detect_regime_change(arm, pred_error):
            # Increase forgetting rate (less forgetting = keep more recent data)
            # Actually, higher forget rate means we keep more history
            # We want to FORGET MORE when regime changes, so DECREASE the multiplier
            self.forget_rates[arm] = max(0.95, self.forget_rates[arm] - 0.01)
        else:
            # Slowly recover toward base forgetting rate
            self.forget_rates[arm] = min(self.base_forget, self.forget_rates[arm] + 0.002)
        
        # Update with adaptive forgetting
        f = self.forget_rates[arm]
        self.A[arm] = f * self.A[arm] + np.outer(context, context)
        self.b[arm] = f * self.b[arm] + reward * context
        
        # Update inverse using Sherman-Morrison or just recompute (dim is small)
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Fallback: add small regularization
            self.A[arm] += 1e-6 * np.eye(self.d)
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]
# EVOLVE-BLOCK-END
