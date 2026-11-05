# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling 
    with adaptive exponential forgetting.
    
    Uses Sherman-Morrison updates for efficient inverse maintenance.
    Adapts the forgetting factor gamma per arm based on recent prediction errors.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize precision matrices A and vectors b
        # Start with a moderate ridge for stability
        lambda_init = 0.1
        self.A = [np.eye(dim) * lambda_init for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Maintain inverses for Thompson Sampling and fast updates
        self.A_inv = [np.eye(dim) / lambda_init for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factors per arm
        # Start with a moderate forgetting rate
        self.gammas = [0.90] * n_arms
        
        # Exploration is handled by Thompson Sampling, but we can add a small 
        # epsilon-random component in the very beginning to ensure all arms 
        # are visited at least once to get initial estimates.
        self.round_count = 0
        self.visits = [0] * n_arms
        
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit
        
        # For adaptive gamma: track recent squared errors
        self.recent_errors = [0.0] * n_arms
        self.error_ema_alpha = 0.2 # How fast the error EMA updates

    def _update_arm(self, arm, context, reward):
        """
        Update the linear model for a specific arm using Sherman-Morrison.
        """
        x = context
        g = self.gammas[arm]
        
        # 1. Compute innovation (prediction error) before updating
        pred = self.theta[arm] @ x
        innovation = reward - pred
        innov_sq = innovation * innovation
        
        # 2. Adapt forgetting factor gamma
        # Update EMA of squared error
        self.recent_errors[arm] = (1 - self.error_ema_alpha) * self.recent_errors[arm] + self.error_ema_alpha * innov_sq
        
        # Map error to gamma.
        # High error -> Low gamma (fast forgetting)
        # Low error -> High gamma (slow forgetting, stability)
        # We want gamma in [0.7, 0.99]
        err_scale = np.sqrt(self.recent_errors[arm])
        
        # Tanh scaling: 
        # If err_scale is 0, tanh(0)=0 -> gamma = 0.99
        # If err_scale is large, tanh(inf)=1 -> gamma = 0.70
        # We scale the input to tanh to control sensitivity
        target_gamma = 0.99 - 0.29 * np.tanh(err_scale / 0.5)
        
        # Smoothly transition gamma to avoid oscillations
        self.gammas[arm] = 0.8 * self.gammas[arm] + 0.2 * target_gamma
        self.gammas[arm] = np.clip(self.gammas[arm], 0.70, 0.99)
        
        # 3. Update A, b, A_inv, theta
        # A_new = g * A_old + x x^T
        # b_new = g * b_old + r x
        
        # Sherman-Morrison for A_inv:
        # Let A_old_inv be the current inverse.
        # We are effectively updating (g * A_old + x x^T).
        # This is not a standard rank-1 update on A_old_inv directly because of the scaling g.
        # 
        # Standard EW-RL update for inverse:
        # P_{t+1} = (1/g) * (P_t - (P_t x) (P_t x)^T / (g + x^T P_t x))
        
        P = self.A_inv[arm]
        Px = P @ x
        
        denom = g + x @ Px
        
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse (rare)
            self.A[arm] = g * self.A[arm] + np.outer(x, x)
            self.b[arm] = g * self.b[arm] + reward * x
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        new_P = (P - np.outer(Px, Px) / denom) / g
        
        # Update b
        self.b[arm] = g * self.b[arm] + reward * x
        
        # Update theta
        self.theta[arm] = new_P @ self.b[arm]
        
        # Update stored inverse
        self.A_inv[arm] = new_P
        
        # Update A for potential fallback/debugging (optional, but good for consistency)
        self.A[arm] = g * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return a deterministic arm based on context to finish quickly
            return int(np.argmax(context)) % self.n
            
        context = np.asarray(context, dtype=np.float64)
        
        # Initial exploration: ensure all arms are visited at least once
        # This prevents zero-visit arms from having undefined or biased samples
        if self.round_count <= self.n:
            # Random selection in the first few rounds
            return int(np.random.randint(self.n))
            
        # Thompson Sampling
        # Sample theta_tilde ~ N(theta, A_inv) for each arm
        best_arm = -1
        best_score = -np.inf
        
        for a in range(self.n):
            theta = self.theta[a]
            cov = self.A_inv[a]
            
            # Ensure covariance is positive definite (add small jitter if needed)
            # In practice, Sherman-Morrison with ridge should keep it PD, 
            # but numerical errors can creep in.
            # We sample by: theta_tilde = theta + L @ z, where cov = L L^T
            # To avoid Cholesky decomposition cost, we can use a simpler approximation 
            # or just sample from multivariate normal.
            # np.random.multivariate_normal is available but can be slow if done many times.
            # Since dim is small (likely < 10) and n_arms is 5, it's fine.
            
            try:
                # Sample from N(theta, cov)
                theta_tilde = np.random.multivariate_normal(theta, cov)
                score = theta_tilde @ context
            except Exception:
                # Fallback if covariance is not PD
                score = theta @ context + np.random.normal(0, 1)
                
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        self.visits[arm] += 1
        
        self._update_arm(arm, context, reward)
# EVOLVE-BLOCK-END
