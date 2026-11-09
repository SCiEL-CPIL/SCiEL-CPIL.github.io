import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy using Thompson Sampling.
    
    Strategy:
    1. Per-arm Bayesian Linear Regression models (same as current).
    2. Adaptive forgetting factor based on local and global prediction error (same as current).
    3. Action Selection via Thompson Sampling: Instead of UCB (mean + alpha * std), 
       we sample a theta from the posterior distribution for each arm and pick the 
       arm with the highest sampled mean reward.
    
    This structural change (UCB -> TS) removes the sensitivity to the exploration 
    parameter 'alpha' and allows the inherent uncertainty (variance) to drive 
    exploration more organically. In non-stationary settings, when the model 
    becomes stale (variance increases due to decay), TS will naturally explore 
    more, whereas UCB might under-explore if alpha is fixed too low.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize A with regularization
        # Using identity matrix with a slightly higher value for initial stability
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
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Random state for Thompson Sampling
        # Using a fixed seed for reproducibility in evaluation, but distinct from global
        self.rng = np.random.default_rng(42)

    def _get_theta_and_cov(self, A, b):
        """
        Compute theta = A^-1 b and the covariance matrix A^-1.
        For Thompson Sampling, we need to sample from N(theta, A^-1).
        """
        try:
            theta = np.linalg.solve(A, b)
            # We need the inverse of A for the covariance.
            # Since A is symmetric positive definite (sum of outer products + lambda*I),
            # we can use np.linalg.inv. For dim=5, this is very fast.
            A_inv = np.linalg.inv(A)
            return theta, A_inv
        except np.linalg.LinAlgError:
            return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
            
        context = np.asarray(context)
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n):
            try:
                theta, A_inv = self._get_theta_and_cov(self.A[a], self.b[a])
                
                # Sample a theta from the posterior N(theta, A_inv)
                # theta_sampled ~ N(theta, A_inv)
                # We can do: theta + L @ z, where A_inv = L L^T (Cholesky)
                # Or simply use multivariate_normal.
                # For dim=5, multivariate_normal is fine.
                theta_sampled = self.rng.multivariate_normal(theta, A_inv)
                
                # Compute expected reward for this arm with the sampled theta
                score = theta_sampled @ context
                
                if score > best_score:
                    best_score = score
                    best_arm = a
                    
            except Exception:
                # If sampling fails, fall back to mean
                try:
                    theta, _ = self._get_theta_and_cov(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except Exception:
                    continue

        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
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
        err_norm_local = min(max(self.error_ema[arm], 0.0), 1.0)
        local_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_local
        
        err_norm_global = min(max(self.global_error_ema, 0.0), 1.0)
        global_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_global
        
        adaptive_alpha = min(local_alpha, global_alpha)
        adaptive_alpha = np.clip(adaptive_alpha, self.min_alpha, self.max_alpha)

        # 5. Update Model for the selected arm
        self.A[arm] = adaptive_alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = adaptive_alpha * self.b[arm] + reward * context
        
        # 6. Global Update for Other Arms
        if self.global_error_ema > 0.5:
            global_decay = 1.0 - 0.05 * self.global_error_ema
            for a in range(self.n):
                if a != arm:
                    self.A[a] = global_decay * self.A[a]
                    self.b[a] = global_decay * self.b[a]

        # 7. Time guard
        if time.time() > self.deadline:
            pass
