import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Strategy:
    Uses per-arm Bayesian Linear Regression models with an adaptive forgetting factor.
    Instead of a fixed decay (e.g., 0.995), the decay rate is modulated by the 
    recent prediction error (surprise) of the selected arm. High error triggers 
    faster forgetting (lower alpha) to adapt to changes; low error triggers slower 
    forgetting (higher alpha) to retain knowledge.
    
    Improvement:
    Adds a Global Non-stationarity Detector. If the global prediction error is high,
    it forces a faster forgetting rate across all arms, preventing stale models in
    unpulled arms from persisting.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_alpha = 0.990  # Base forgetting factor (slower decay)
        self.max_alpha = 0.999   # Max forgetting factor (very slow decay when stable)
        self.min_alpha = 0.900   # Min forgetting factor (fast decay when unstable)
        
        # State for adaptive control
        self.error_history = [0.0 for _ in range(n_arms)]
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.error_ema_alpha = 0.1  # Smoothing for error EMA
        
        # Global state
        self.global_error_ema = 0.0
        self.global_error_ema_alpha = 0.2
        
        # Exploration parameter
        self.alpha_ucb = 1.2 # Slightly increased exploration
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0 # Safety margin for 120s limit

    def _get_inv(self, A):
        """
        Compute inverse of A. 
        For small dimensions (dim=5), explicit inverse is fast and allows 
        direct use in UCB term calculation. 
        We use solve for theta but inv for the UCB variance term to avoid 
        repeated solves in the scoring loop if we were batching, but here 
        per-arm inv is cheap. 
        Actually, for dim=5, np.linalg.inv is O(5^3)=125 ops, very fast.
        """
        return np.linalg.inv(A)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        scores = []
        context = np.asarray(context)
        
        for a in range(self.n):
            try:
                inv_A = self._get_inv(self.A[a])
                theta = inv_A @ self.b[a]
                
                # Mean prediction
                mean = theta @ context
                
                # Variance term
                variance = context @ inv_A @ context
                if variance < 0:
                    variance = 0.0
                
                # UCB score
                # If global error is high, we might want to explore more
                # Let's dynamically adjust alpha_ucb slightly
                # High global error -> higher alpha_ucb
                dynamic_alpha = self.alpha_ucb + 0.5 * min(self.global_error_ema, 1.0)
                
                score = mean + dynamic_alpha * np.sqrt(variance)
                scores.append(score)
            except Exception:
                # Fallback if numerical issues
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            inv_A = self._get_inv(self.A[arm])
            theta = inv_A @ self.b[arm]
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_abs = abs(error)
        except Exception:
            error_abs = 1.0

        # 2. Update Local Error EMA
        self.error_ema[arm] = (1 - self.error_ema_alpha) * self.error_ema[arm] + self.error_ema_alpha * error_abs
        
        # 3. Update Global Error EMA
        self.global_error_ema = (1 - self.global_error_ema_alpha) * self.global_error_ema + self.global_error_ema_alpha * error_abs

        # 4. Determine adaptive forgetting factor
        # We combine local and global signals.
        # Local signal: If this arm is wrong, forget it fast.
        # Global signal: If the environment is changing (global error high), forget ALL arms faster.
        
        # Local alpha calculation
        err_norm_local = min(max(self.error_ema[arm], 0.0), 1.0)
        local_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_local
        
        # Global alpha calculation
        # If global error is high, we want a lower alpha (faster forgetting) for everyone.
        # Map global error to a multiplier or a target alpha.
        # Let's say if global_error_ema is 0, global_alpha = max_alpha.
        # If global_error_ema is 1.0, global_alpha = min_alpha.
        err_norm_global = min(max(self.global_error_ema, 0.0), 1.0)
        global_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_global
        
        # Combine: The effective alpha for the current arm is the minimum of the two?
        # No, we want the *fastest* forgetting if either signal is strong.
        # So we take the minimum alpha (which corresponds to fastest forgetting).
        adaptive_alpha = min(local_alpha, global_alpha)
        
        # Ensure alpha is within bounds
        adaptive_alpha = np.clip(adaptive_alpha, self.min_alpha, self.max_alpha)

        # 5. Update Model
        # A = alpha * A + xx^T
        # b = alpha * b + x * r
        self.A[arm] = adaptive_alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = adaptive_alpha * self.b[arm] + reward * context
        
        # 6. Global Update for Other Arms?
        # If global error is high, it suggests the environment has changed.
        # We should also apply a slight decay to the other arms to prevent them
        # from holding onto stale weights that are no longer relevant.
        # This is a heuristic: apply a "global decay" to unpulled arms.
        # To avoid excessive computation, we only do this if global error is significant.
        if self.global_error_ema > 0.5:
            # Apply a decay factor to all other arms
            # Use a slightly more aggressive decay for unobserved arms if global drift is detected
            # This helps "reset" the exploration of other arms.
            for a in range(self.n):
                if a != arm:
                    # Decay their A and b slightly
                    # Use a decay factor close to 1.0 but less, e.g., 0.99
                    # This is conservative to avoid destroying good estimates.
                    # Let's use a factor derived from global error.
                    # If global error is 1.0, decay by 0.95. If 0.5, decay by 0.99.
                    global_decay = 1.0 - 0.05 * self.global_error_ema
                    self.A[a] = global_decay * self.A[a]
                    self.b[a] = global_decay * self.b[a]

        # 7. Time guard
        if time.time() > self.deadline:
            # We can't really do much here other than ensure future calls are fast
            pass
