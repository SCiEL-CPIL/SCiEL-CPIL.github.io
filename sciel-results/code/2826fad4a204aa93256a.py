"""
Non-stationary Linear Contextual Bandit Policy.

Uses an adaptive Linear UCB approach with:
1. Recursive Least Squares (RLS) for parameter estimation.
2. Per-arm adaptive forgetting factors based on recent prediction errors.
3. "Shock" reset mechanism for arms that exhibit large prediction mismatches,
   allowing rapid recovery from regime changes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Model parameters for each arm
        # A: d x d matrix, b: d vector
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # Regularization start
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm forgetting factor parameters
        self.lambda_base = 0.99
        self.lambda_min = 0.80
        self.lambda_max = 0.999
        self.current_lambda = [self.lambda_base] * n_arms
        
        # Per-arm error tracking for adaptive lambda
        self.error_buffer = [np.zeros(50) for _ in range(n_arms)]
        self.error_idx = [0] * n_arms
        self.error_count = [0] * n_arms
        self.expected_var = [1.0] * n_arms  # Per-arm noise variance estimates
        
        # Exploration parameter
        self.alpha_start = 2.5
        self.alpha_min = 0.8
        self.decay_rate = 0.0005
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0 # 5s margin
        
        # Reset threshold multiplier
        self.reset_threshold = 3.0

    def _solve(self, A, b):
        """Solve linear system with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_inverse_solve(self, A, vec):
        """Solve A x = vec."""
        try:
            return np.linalg.solve(A, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, vec, rcond=None)[0]

    def _update_lambda(self, arm, sq_error):
        """Update per-arm forgetting factor based on error."""
        # Store error
        self.error_buffer[arm][self.error_idx[arm]] = sq_error
        self.error_idx[arm] = (self.error_idx[arm] + 1) % len(self.error_buffer[arm])
        if self.error_count[arm] < len(self.error_buffer[arm]):
            self.error_count[arm] += 1
            
        if self.error_count[arm] == 0:
            return
            
        # Calculate average recent squared error
        recent_errors = self.error_buffer[arm][:self.error_count[arm]]
        avg_sq_err = np.mean(recent_errors)
        
        # Adaptive lambda:
        # If avg_sq_err is high, we want low lambda (more forgetting).
        # If avg_sq_err is low, we want high lambda (more memory).
        # Map avg_sq_err to [0, 1] range using a scaling factor.
        error_ratio = avg_sq_err / max(self.expected_var[arm], 0.1)
        
        # Use a soft saturation function
        penalty = error_ratio / (1.0 + error_ratio)
        
        new_lambda = self.lambda_max - (self.lambda_max - self.lambda_min) * penalty
        
        # Smooth the lambda update to avoid oscillation
        self.current_lambda[arm] = 0.85 * self.current_lambda[arm] + 0.15 * new_lambda
        
        # Update expected variance estimate (slowly)
        self.expected_var[arm] = 0.98 * self.expected_var[arm] + 0.02 * (avg_sq_err + 0.1)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
             # Exploit best known arm
             best_arm = 0
             best_score = -np.inf
             for a in range(self.n_arms):
                 theta = self._solve(self.A[a], self.b[a])
                 mu = theta @ context
                 if mu > best_score:
                     best_score = mu
                     best_arm = a
             return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha with non-stationarity-aware floor
        # In non-stationary settings, maintain more exploration
        base_alpha = self.alpha_start * np.exp(-self.decay_rate * self.t)
        # Add a component that increases if we've seen many resets (indicates high non-stationarity)
        alpha = max(self.alpha_min, base_alpha)
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            A = self.A[a]
            b = self.b[a]
            
            theta = self._solve(A, b)
            mu = theta @ context
            
            # Uncertainty term: x^T A^{-1} x
            A_inv_x = self._get_inverse_solve(A, context)
            v = context @ A_inv_x
            
            # UCB Score
            score = mu + alpha * np.sqrt(max(v, 0.0))
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction and residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        
        # Uncertainty for reset check
        A_inv_x = self._get_inverse_solve(A, context)
        v = context @ A_inv_x
        std_pred = np.sqrt(max(v, 0.0)) + np.sqrt(max(self.expected_var[arm], 0.1))
        
        residual = reward - mu_pred
        sq_error = residual ** 2
        
        # 2. Update Per-Arm Lambda
        self._update_lambda(arm, sq_error)
        lam = self.current_lambda[arm]
        
        # 3. Check for Shock Reset
        # If residual is significantly larger than expected standard deviation
        if abs(residual) > self.reset_threshold * std_pred:
            # Reset model for this arm to prioritize the new data point
            self.A[arm] = np.eye(self.dim) * 1.0
            self.b[arm] = reward * context
        else:
            # Standard RLS update with forgetting
            self.A[arm] = lam * self.A[arm] + np.outer(context, context)
            self.b[arm] = lam * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
