"""Non-stationary linear contextual bandit with adaptive Discounted Least Squares (DLS) and residual-driven forgetting factor."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Discounted Least Squares (DLS).
    
    Structural Change:
    - Moves from Kalman Filter (State-Space) to Recursive Least Squares with Forgetting Factor.
    - This simplifies the covariance tracking to a weighted Hessian inverse.
    - Adaptation is controlled by a dynamic forgetting factor (gamma) instead of process noise Q.
    
    Adaptation Mechanism (from Inspirations):
    - Uses a Tanh-saturated EMA of squared residuals to estimate drift.
    - High drift -> Lower gamma (faster forgetting, more responsive).
    - Low drift -> Higher gamma (slower forgetting, more stable).
    
    Exploration:
    - UCB with count-based alpha decay.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 130.0  # Safety margin

        # DLS State: A (weighted Hessian) and b (weighted gradient)
        # Initialize with small ridge for stability
        self.ridge = 1.0
        self.A = [self.ridge * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Store inverse of A for fast variance calculation in select
        # Since dim is small (5), we can recompute or update. 
        # For efficiency in select, we might just compute A_inv on the fly or cache it.
        # Caching A_inv is risky if A changes. But A only changes in update.
        # We can update A_inv using Sherman-Morrison-Woodbury in update, 
        # or just recompute A_inv = np.linalg.inv(A) in select if needed.
        # Given d=5, np.linalg.inv is ~microseconds. Let's recompute in select for safety/simplicity.
        
        # Adaptive Forgetting Factor (gamma)
        # gamma in [0.90, 0.99]
        self.gamma_min = 0.90
        self.gamma_max = 0.99
        self.gamma = 0.95  # Initial guess
        
        # Residual tracking for adaptive gamma
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1  # Smoothing for residual EMA
        
        # Exploration
        self.alpha_base = 1.2
        self.alpha_decay = 20.0
        
        # Deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self, arm):
        """Count-based exploration decay."""
        c = self.count[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def _update_gamma(self, arm, residual_sq):
        """
        Update the forgetting factor based on the observed residual.
        Uses tanh saturation for smooth, bounded adaptation.
        """
        # Update EMA of squared residual
        self.residual_ema[arm] = (1 - self.ema_alpha) * self.residual_ema[arm] + self.ema_alpha * residual_sq
        
        # Map EMA to gamma
        # If residual_ema is high (drift), we want lower gamma (faster forgetting).
        # If residual_ema is low (stable), we want higher gamma (slower forgetting).
        
        # Scale the residual to control sensitivity
        # tanh(x) goes from -1 to 1. We want a value in [0, 1] representing "drift amount".
        # Let's use sqrt of EMA to get back to magnitude scale
        drift_metric = np.sqrt(max(self.residual_ema[arm], 0.0))
        
        # Normalize drift metric. Assume typical rewards are O(1).
        # If drift is > 1.0, consider it high.
        normalized_drift = drift_metric / 1.0
        
        # tanh(normalized_drift) -> [0, 1) for positive inputs
        # 0 means no drift -> gamma_max
        # 1 means high drift -> gamma_min
        sat = np.tanh(normalized_drift)
        
        # Interpolate between gamma_min and gamma_max
        # gamma = gamma_max - (gamma_max - gamma_min) * sat
        self.gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * sat

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            # Greedy fallback: compute theta = A^-1 b for all arms
            scores = []
            for a in range(self.n):
                try:
                    A_inv = np.linalg.inv(self.A[a])
                    theta = A_inv @ self.b[a]
                    mean = theta @ context
                    scores.append(mean)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        # Pre-compute context outer product for potential reuse? 
        # No, A is per arm.
        
        for a in range(self.n):
            # Compute theta and variance
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with ridge)
                A_inv = np.eye(self.d) / self.ridge
                
            theta = A_inv @ self.b[a]
            mean = theta @ context
            
            # Variance: x^T A^{-1} x
            var = context @ A_inv @ context
            var = max(var, 0.0)
            
            alpha_a = self._get_alpha(a)
            ucb = mean + alpha_a * np.sqrt(var)
            scores.append(ucb)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        a = arm
        self.t += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # 1. Predict/Estimate current theta to calculate residual
        # We use the current A and b to estimate theta before update
        try:
            A_inv = np.linalg.inv(self.A[a])
            theta_old = A_inv @ self.b[a]
        except np.linalg.LinAlgError:
            theta_old = np.zeros(self.d)

        # 2. Calculate Residual
        predicted_reward = theta_old @ context
        residual = reward - predicted_reward
        residual_sq = residual * residual

        # 3. Update Adaptive Gamma
        self._update_gamma(a, residual_sq)
        gamma = self.gamma

        # 4. DLS Update
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + x * r
        
        self.A[a] = gamma * self.A[a] + np.outer(context, context)
        self.b[a] = gamma * self.b[a] + context * reward

        self.count[a] += 1
# EVOLVE-BLOCK-END
