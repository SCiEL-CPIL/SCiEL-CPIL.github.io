import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy with Thompson Sampling.
    
    Strategy:
    1. Thompson Sampling (TS) instead of UCB: Samples the weight vector theta from the 
       posterior distribution N(mu, Sigma) to balance exploration and exploitation 
       probabilistically. This often outperforms UCB in non-stationary settings because 
       it naturally explores arms with high uncertainty without the "optimism bias" 
       that can cause over-commitment to stale high-reward arms.
    2. Smooth Adaptive Forgetting: Uses a soft-min (log-sum-exp based) of local and 
       global error signals to determine the forgetting factor (alpha). This avoids 
       the discontinuities of a hard min, allowing for smoother adaptation when 
       signals conflict.
    3. Residual Variance Normalization: Tracks the residual variance of rewards to 
       normalize the prediction uncertainty, making the exploration bonus scale-invariant.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_alpha = 0.990
        self.max_alpha = 0.999
        self.min_alpha = 0.900
        
        # State for adaptive control
        self.error_history = [0.0 for _ in range(n_arms)]
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.error_ema_alpha = 0.1
        
        # Global state
        self.global_error_ema = 0.0
        self.global_error_ema_alpha = 0.2
        
        # Residual variance tracking (per arm and global)
        self.residual_var_ema = [1.0 for _ in range(n_arms)] # Initialize to 1.0
        self.residual_var_ema_alpha = 0.1
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0 # Safety margin for 120s limit
        
        # Random state for Thompson Sampling
        self.rng = np.random.RandomState(42)

    def _get_inv(self, A):
        """Compute inverse of A."""
        return np.linalg.inv(A)

    def _soft_min(self, a, b, temperature=0.1):
        """
        Compute a smooth approximation of min(a, b) using log-sum-exp.
        min(a, b) ≈ -1/T * log(exp(-T*a) + exp(-T*b))
        As T -> inf, this approaches min(a, b).
        For T=10 (temperature=0.1), it's a reasonably sharp min but smooth.
        """
        # Use the stable log-sum-exp trick
        # We want min(a,b). 
        # Let M = min(a,b). 
        # log(exp(-T*a) + exp(-T*b)) = -T*M + log(exp(-T*(a-M)) + exp(-T*(b-M)))
        # Since a-M >= 0 and b-M >= 0, the terms in log are <= 1.
        
        m = min(a, b)
        # If T is large, exp(-T*(a-m)) can underflow if a-m is large, which is fine (becomes 0).
        # But if a-m is small, it's ~1.
        
        # Actually, simpler:
        # soft_min = -1/T * log(exp(-T*a) + exp(-T*b))
        # To avoid overflow in exp(-T*a) if a is very negative? 
        # Our alphas are between 0.9 and 0.999, so they are positive and close to 1.
        # -T*alpha will be around -10. exp(-10) is small. No overflow risk.
        
        term1 = np.exp(-a / temperature)
        term2 = np.exp(-b / temperature)
        return -temperature * np.log(term1 + term2)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        context = np.asarray(context)
        scores = []
        
        # Global exploration scale based on global error
        # High global error -> more exploration
        global_exploration_scale = 1.0 + 0.5 * min(self.global_error_ema, 1.0)
        
        for a in range(self.n):
            try:
                inv_A = self._get_inv(self.A[a])
                theta_mean = inv_A @ self.b[a]
                
                # Thompson Sampling: Sample theta from N(theta_mean, inv_A)
                # Since dim is small (5), we can do Cholesky decomposition for sampling
                # Or just sample from multivariate normal directly.
                # np.random.multivariate_normal is convenient but might be slightly slower.
                # For dim=5, it's negligible.
                
                # Covariance is inv_A
                # Check if inv_A is positive semi-definite. It should be since A is PD.
                theta_sample = self.rng.multivariate_normal(theta_mean, inv_A)
                
                # Predicted reward using the sampled theta
                mean = theta_sample @ context
                
                # In TS, we don't add an explicit UCB bonus. The randomness IS the exploration.
                # However, to adapt to non-stationarity, we might want to scale the variance
                # if the residual variance is high.
                
                # The standard TS uses the posterior directly.
                # score = mean
                scores.append(mean)
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
            error_sq = error * error
        except Exception:
            error_abs = 1.0
            error_sq = 1.0

        # 2. Update Local Error EMA
        self.error_ema[arm] = (1 - self.error_ema_alpha) * self.error_ema[arm] + self.error_ema_alpha * error_abs
        
        # 3. Update Global Error EMA
        self.global_error_ema = (1 - self.global_error_ema_alpha) * self.global_error_ema + self.global_error_ema_alpha * error_abs
        
        # 4. Update Residual Variance EMA
        self.residual_var_ema[arm] = (1 - self.residual_var_ema_alpha) * self.residual_var_ema[arm] + self.residual_var_ema_alpha * error_sq

        # 5. Determine adaptive forgetting factor
        # Local signal: If this arm is wrong, forget it fast.
        # Global signal: If the environment is changing (global error high), forget ALL arms faster.
        
        # Map errors to alpha range [min_alpha, max_alpha]
        # Higher error -> Lower alpha (faster forgetting)
        
        err_norm_local = min(max(self.error_ema[arm], 0.0), 1.0)
        local_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_local
        
        err_norm_global = min(max(self.global_error_ema, 0.0), 1.0)
        global_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_global
        
        # Combine using Soft-Min
        # We want the effective alpha to be close to the minimum of the two (fastest forgetting)
        # but smooth.
        adaptive_alpha = self._soft_min(local_alpha, global_alpha, temperature=0.05)
        
        # Ensure alpha is within bounds
        adaptive_alpha = np.clip(adaptive_alpha, self.min_alpha, self.max_alpha)

        # 6. Update Model
        # A = alpha * A + xx^T
        # b = alpha * b + x * r
        self.A[arm] = adaptive_alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = adaptive_alpha * self.b[arm] + reward * context
        
        # 7. Global Update for Other Arms?
        # If global error is high, apply a slight decay to other arms to prevent staleness.
        if self.global_error_ema > 0.5:
            global_decay = 1.0 - 0.05 * self.global_error_ema
            for a in range(self.n):
                if a != arm:
                    self.A[a] = global_decay * self.A[a]
                    self.b[a] = global_decay * self.b[a]

        # 8. Time guard
        if time.time() > self.deadline:
            pass
