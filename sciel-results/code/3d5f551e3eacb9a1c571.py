"""Non-stationary linear contextual bandit with adaptive forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    LinUCB with adaptive forgetting factor and adaptive exploration.
    
    - Forgetting factor decreases (faster forgetting) when prediction error is high,
      to track non-stationary weight shifts.
    - Exploration coefficient increases when volatility is high.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive parameters
        self.base_gamma = 0.99       # base forgetting factor
        self.min_gamma = 0.90        # minimum (fastest forgetting)
        self.base_alpha = 1.0        # base exploration
        self.max_alpha = 3.0         # max exploration
        
        # Error tracking
        self.ema_error = 0.0         # exponential moving average of squared prediction error
        self.error_beta = 0.95       # EMA decay for error tracking
        self.n_obs = 0
        self._start_time = time.time()
        self._deadline = 115.0       # safety margin before 120s limit

    def _get_adaptive_params(self):
        """Compute adaptive gamma and alpha based on recent prediction error."""
        # Clamp error to avoid extreme values
        err = min(self.ema_error, 10.0)
        
        # Forgetting factor: lower error -> closer to base_gamma (slow forgetting)
        # Higher error -> lower gamma (faster forgetting)
        gamma = self.base_gamma - 0.05 * err
        gamma = max(self.min_gamma, min(self.base_gamma, gamma))
        
        # Exploration: higher error -> more exploration
        alpha = self.base_alpha + 0.5 * err
        alpha = min(self.max_alpha, alpha)
        
        return gamma, alpha

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: if close to timeout, use pure exploitation
        if time.time() - self._start_time > self._deadline:
            scores = []
            for a in range(self.n):
                theta = np.linalg.solve(self.A[a], self.b[a])
                scores.append(theta @ context)
            return int(np.argmax(scores))
        
        gamma, alpha = self._get_adaptive_params()
        
        scores = []
        for a in range(self.n):
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                pred = theta @ context
                # UCB bonus via Cholesky
                L = np.linalg.cholesky(self.A[a])
                # Solve L y = context, then bonus = alpha * ||y||
                y = np.linalg.solve(L, context)
                bonus = alpha * np.sqrt(np.dot(y, y))
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                theta = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]
                pred = theta @ context
                inv_A = np.linalg.pinv(self.A[a])
                bonus = alpha * np.sqrt(context @ inv_A @ context)
                scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.n_obs += 1
        
        # Compute prediction error for this arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta @ context
        except np.linalg.LinAlgError:
            pred = 0.0
        squared_error = (reward - pred) ** 2
        
        # Update EMA of error
        self.ema_error = self.error_beta * self.ema_error + (1 - self.error_beta) * squared_error
        
        # Get adaptive forgetting factor
        gamma, _ = self._get_adaptive_params()
        
        # Update A and b with forgetting
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
