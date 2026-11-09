"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy with Per-Arm Adaptive Forgetting.
    
    Uses LinUCB with:
    1. Per-arm adaptive forgetting factor (gamma) based on residual error magnitude.
    2. Sherman-Morrison formula for efficient inverse updates.
    3. Adaptive exploration parameter (beta) that increases with uncertainty/error.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # LinUCB parameters
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive Forgetting parameters
        self.gamma = [0.995] * n_arms  # Initial forgetting factor
        self.gamma_min = 0.90          # Fastest forgetting
        self.gamma_max = 0.999         # Slowest forgetting
        self.gamma_decay = 0.95        # How fast gamma drops when change detected
        self.gamma_grow = 1.01         # How fast gamma recovers when stable
        
        # Volatility / Change Detection
        self.ema_reward = [0.0] * n_arms
        self.ema_res_sq = [1.0] * n_arms  # Initialize with 1 to avoid division by zero initially
        self.ema_alpha = 0.5              # Fast EMA for change detection
        
        # Exploration
        self.beta_base = 1.5
        self.beta = [self.beta_base] * n_arms
        self.beta_max = 5.0
        self.beta_decay = 0.9             # Beta decays back to base
        self.beta_increase = 1.2          # Beta increases on large residual
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def select(self, context):
        # Deadline check: if we are close to the limit, just pick the best known arm
        # to avoid any potential overhead, though select is O(n*d^2) which is tiny.
        if time.time() > self.deadline:
             return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))

        context = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # theta = inv_A @ b is already stored
            pred = self.theta[a] @ context
            # Uncertainty = sqrt(x^T A^-1 x)
            unc = np.sqrt(context @ self.inv_A[a] @ context)
            
            scores[a] = pred + self.beta[a] * unc
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.array(context)
        
        # 1. Calculate Residual
        pred = self.theta[arm] @ context
        residual = reward - pred
        res_sq = residual * residual
        
        # 2. Update Volatility Estimates (for change detection)
        # Update EMA of rewards (optional, but good for baseline)
        self.ema_reward[arm] = self.ema_alpha * reward + (1 - self.ema_alpha) * self.ema_reward[arm]
        
        # Update EMA of squared residuals
        self.ema_res_sq[arm] = self.ema_alpha * res_sq + (1 - self.ema_alpha) * self.ema_res_sq[arm]
        
        # 3. Adaptive Forgetting Factor
        # If the current residual is significantly larger than the expected variance,
        # it suggests a change in the underlying function.
        # We use a threshold relative to the current EMA of squared residuals.
        if self.ema_res_sq[arm] > 1e-8:
            ratio = res_sq / self.ema_res_sq[arm]
        else:
            ratio = 1.0
            
        if ratio > 2.0:  # Significant deviation
            # Drop gamma to forget faster
            self.gamma[arm] = max(self.gamma_min, self.gamma[arm] * self.gamma_decay)
            # Increase exploration
            self.beta[arm] = min(self.beta_max, self.beta[arm] * self.beta_increase)
        else:
            # Stable environment, recover gamma and beta
            self.gamma[arm] = min(self.gamma_max, self.gamma[arm] * self.gamma_grow)
            self.beta[arm] = max(self.beta_base, self.beta[arm] * self.beta_decay)
            
        # 4. Update Linear Model
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + r x
        
        g = self.gamma[arm]
        
        # Update A and b
        self.A[arm] = g * self.A[arm] + np.outer(context, context)
        self.b[arm] = g * self.b[arm] + reward * context
        
        # 5. Update Inverse using Sherman-Morrison
        # We need (g * A_old + x x^T)^-1
        # = (g * (A_old + (1/g) x x^T))^-1
        # = (1/g) * (A_old + (1/g) x x^T)^-1
        # Let M = A_old, u = (1/sqrt(g)) * x, v = (1/sqrt(g)) * x
        # (M + u v^T)^-1 = M^-1 - (M^-1 u v^T M^-1) / (1 + v^T M^-1 u)
        
        # Note: self.inv_A[arm] is currently A_old^-1
        inv_A_old = self.inv_A[arm]
        
        u = context / np.sqrt(g)
        
        # Calculate denominator: 1 + v^T M^-1 u = 1 + u^T M^-1 u (since u=v)
        tmp = inv_A_old @ u
        denom = 1.0 + u @ tmp
        
        if denom < 1e-8:
             # Numerical instability fallback: recompute inverse
             self.inv_A[arm] = np.linalg.inv(self.A[arm])
        else:
             # Sherman-Morrison update
             # inv_A_new_unscaled = inv_A_old - (inv_A_old @ u @ u^T @ inv_A_old) / denom
             # Then multiply by 1/g
             # Let's compute the term (inv_A_old @ u) @ (u^T @ inv_A_old)
             # We already have tmp = inv_A_old @ u
             # So term = tmp @ tmp.T
             
             update_term = (tmp @ tmp.T) / denom
             inv_A_new_unscaled = inv_A_old - update_term
             self.inv_A[arm] = inv_A_new_unscaled / g
             
        # 6. Update Theta
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

# EVOLVE-BLOCK-END
