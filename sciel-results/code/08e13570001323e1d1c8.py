"""Differential Linear Contextual Bandit with Global Drift Compensation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy using Differential Modeling.
    
    Strategy:
    - Models the non-stationarity as a global linear trend affecting all arms.
    - Maintains a global weight vector (w_global) estimated via RLS on all (context, reward) pairs.
    - For each arm, maintains a LinUCB model on the 'residual' reward (r - w_global . x).
    - This isolates the arm-specific preference from the common environmental shift.
    - Selection is based on the expected reward of the arm, adjusted by the global trend.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global drift estimator (RLS)
        # We use a small initial covariance to avoid huge initial swings, 
        # but large enough to learn.
        self.w_global = np.zeros(dim)
        self.P_global = np.eye(dim) * 100.0
        self.alpha_global = 1.0  # RLS learning rate (inverse forgetting)
        self.forgetting_factor = 0.99  # Slow adaptation for global trend
        
        # Per-arm LinUCB models
        self.means = [np.zeros(dim) for _ in range(n_arms)]
        self.covs = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.ucb_beta = 2.0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            # The model predicts the residual. 
            # The true reward is approx by w_global . x + theta_a . x
            # So expected reward = (w_global + theta_a) . x
            # Uncertainty comes only from the arm-specific part (theta_a)
            # because w_global is shared and its uncertainty is handled globally,
            # but for UCB we usually add the uncertainty of the prediction.
            # Here we assume the global trend is learned well enough that 
            # the arm-specific uncertainty dominates exploration needs.
            
            mean_pred = (self.w_global + self.means[a]) @ context
            
            # Uncertainty from the arm-specific model
            var = context @ self.covs[a] @ context
            std = np.sqrt(var)
            
            # UCB score
            scores[a] = mean_pred + self.ucb_beta * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
            
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Update Global Drift Estimator (RLS)
        # Standard RLS update
        # P = P / lambda - (P x x^T P) / (lambda + x^T P x)
        # w = w + P x (y - w^T x) / (lambda + x^T P x)
        
        lambda_g = self.forgetting_factor
        x = context
        y = reward
        
        Px = self.P_global @ x
        denominator = lambda_g + x @ Px
        
        # Update gain
        K_global = Px / denominator
        
        # Update weight
        innovation_global = y - self.w_global @ x
        self.w_global += K_global * innovation_global
        
        # Update covariance
        self.P_global = (self.P_global - np.outer(K_global, x @ self.P_global)) / lambda_g
        # Symmetrize
        self.P_global = (self.P_global + self.P_global.T) / 2.0
        
        # 2. Update Arm-Specific Model (LinUCB on Residual)
        # Residual reward: r_residual = reward - w_global . context
        # Note: We use the UPDATED w_global for the residual calculation.
        # This effectively projects out the global trend from the reward.
        
        r_residual = reward - self.w_global @ context
        
        # Standard LinUCB update for the selected arm
        # P = P - (P x x^T P) / (1 + x^T P x)
        # w = w + P x (y - w^T x) / (1 + x^T P x)
        
        # For stability, we might use a small regularization or just standard OLS inverse
        # LinUCB typically uses P = (I + X^T X)^-1
        # Update rule:
        # P_new = P_old - (P_old x x^T P_old) / (1 + x^T P_old x)
        
        Px_arm = self.covs[arm] @ x
        denominator_arm = 1.0 + x @ Px_arm
        
        K_arm = Px_arm / denominator_arm
        
        innovation_arm = r_residual - self.means[arm] @ x
        self.means[arm] += K_arm * innovation_arm
        
        self.covs[arm] -= np.outer(K_arm, x @ self.covs[arm])
        # Symmetrize
        self.covs[arm] = (self.covs[arm] + self.covs[arm].T) / 2.0

# EVOLVE-BLOCK-END
