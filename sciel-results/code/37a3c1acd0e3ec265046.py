"""Non-stationary linear contextual bandit with adaptive forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear bandit with adaptive forgetting factor and exploration.
    
    Uses per-arm ridge regression with a dynamically adjusted forgetting factor.
    When recent prediction errors are large, the forgetting factor decreases to
    adapt quickly to non-stationary changes. When errors are small, it increases
    to accumulate more data and reduce variance.
    
    Exploration bonus is also adaptive, scaling with recent error magnitude.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._start_time = time.time()
        self._deadline = 115.0  # safety margin before 120s limit
        
        # Per-arm linear models
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # Regularized covariance
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive parameters
        self.forgetting = 0.95  # Base forgetting factor
        self.alpha = 1.0  # Exploration coefficient
        self.base_alpha = 1.0
        self.base_forgetting = 0.95
        
        # Track recent prediction errors for adaptivity
        self.recent_errors = []
        self.error_window = 20
        self.error_threshold_high = 1.0  # High error -> increase adaptation
        self.error_threshold_low = 0.1   # Low error -> decrease adaptation
        
        # Track per-arm recent rewards for bias estimation
        self.arm_reward_history = [[] for _ in range(n_arms)]
        self.reward_history_len = 10
        
        # Round counter
        self.round = 0

    def _get_adaptive_params(self):
        """Compute adaptive forgetting factor and exploration coefficient."""
        if len(self.recent_errors) < 5:
            return self.base_forgetting, self.base_alpha
        
        recent = self.recent_errors[-self.error_window:]
        mean_err = np.mean(np.abs(recent))
        std_err = np.std(recent)
        
        # Adaptive forgetting: decrease (more forgetting) when errors are high
        # Map mean error to forgetting factor in [0.85, 0.99]
        # High error -> low forgetting (0.85), low error -> high forgetting (0.99)
        if mean_err > self.error_threshold_high:
            # Scale down forgetting
            excess = min(mean_err / self.error_threshold_high, 5.0)
            forgetting = self.base_forgetting - 0.1 * min(excess, 1.0)
            forgetting = max(0.80, min(0.99, forgetting))
        elif mean_err < self.error_threshold_low:
            # Scale up forgetting (retain more history)
            ratio = mean_err / self.error_threshold_low
            forgetting = min(0.995, self.base_forgetting + 0.04 * (1.0 - ratio))
        else:
            # Intermediate: interpolate
            t = (mean_err - self.error_threshold_low) / (self.error_threshold_high - self.error_threshold_low)
            forgetting = self.base_forgetting - 0.05 * t
        
        # Adaptive alpha: increase exploration when errors are high
        if mean_err > self.error_threshold_high:
            excess = min(mean_err / self.error_threshold_high, 3.0)
            alpha = self.base_alpha + 1.0 * min(excess, 2.0)
        elif mean_err < self.error_threshold_low:
            ratio = mean_err / self.error_threshold_low
            alpha = max(0.3, self.base_alpha - 0.5 * (1.0 - ratio))
        else:
            t = (mean_err - self.error_threshold_low) / (self.error_threshold_high - self.error_threshold_low)
            alpha = self.base_alpha + 0.5 * t
        
        # Add small component from std to capture volatility
        alpha += 0.2 * min(std_err / max(mean_err, 1e-8), 1.0)
        
        return forgetting, alpha

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline
        if time.time() - self._start_time > self._deadline:
            # Return arm with highest estimated mean
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    score = float(theta @ context)
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return best_arm
        
        forgetting, alpha = self._get_adaptive_params()
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            try:
                # Solve for theta
                theta = np.linalg.solve(self.A[a], self.b[a])
                mean = float(theta @ context)
                
                # Uncertainty: context^T A^{-1} context
                # Compute via solve for efficiency
                inv_A_ctx = np.linalg.solve(self.A[a], context)
                var = float(context @ inv_A_ctx)
                
                bonus = alpha * np.sqrt(max(var, 0.0))
                scores[a] = mean + bonus
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                try:
                    theta = np.linalg.pinv(self.A[a]) @ self.b[a]
                    mean = float(theta @ context)
                    inv_A = np.linalg.pinv(self.A[a])
                    var = float(context @ inv_A @ context)
                    bonus = alpha * np.sqrt(max(var, 0.0))
                    scores[a] = mean + bonus
                except Exception:
                    scores[a] = 0.0
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.round += 1
        
        # Compute prediction error before update
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            pred = float(theta @ context)
            error = reward - pred
        except np.linalg.LinAlgError:
            error = 0.0
        
        # Track recent errors
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.error_window * 2:
            self.recent_errors = self.recent_errors[-self.error_window * 2:]
        
        # Track per-arm rewards
        self.arm_reward_history[arm].append(reward)
        if len(self.arm_reward_history[arm]) > self.reward_history_len:
            self.arm_reward_history[arm] = self.arm_reward_history[arm][-self.reward_history_len:]
        
        # Update with adaptive forgetting
        forgetting, _ = self._get_adaptive_params()
        
        self.A[arm] = forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = forgetting * self.b[arm] + reward * context
        
        # Periodic re-regularization to prevent numerical drift
        if self.round % 100 == 0:
            for a in range(self.n):
                # Add small regularization to maintain positive definiteness
                self.A[a] += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
