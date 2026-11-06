"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear contextual bandit with a global linear model
    augmented by arm-specific bias terms.
    
    Model: r_t = x_t^T w + b_a + noise
    where w is a shared global weight vector and b_a is an arm-specific bias.
    
    We use a Bayesian linear regression approach with a forgetting factor
    to handle non-stationarity. The shared weight vector allows the policy
    to learn from all arms' observations, improving sample efficiency.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global model: shared weight vector w
        # Using Bayesian linear regression with forgetting factor
        self.A_global = np.eye(dim) * 0.1  # Prior precision (small prior variance)
        self.b_global = np.zeros(dim)
        
        # Arm-specific bias terms
        self.A_bias = [np.array([[1.0]]) for _ in range(n_arms)]
        self.b_bias = [np.array([0.0]) for _ in range(n_arms)]
        
        # Forgetting factor for non-stationarity
        self.forgetting = 0.98
        
        # Exploration parameter
        self.alpha = 1.0

    def select(self, context):
        """
        Select arm using UCB-like exploration.
        
        For each arm a, estimate:
        E[r|context, arm=a] = context^T w_hat + b_hat_a
        Variance = context^T (A_global^{-1}) context + (A_bias[a]^{-1})
        """
        context = np.asarray(context, dtype=np.float64)
        d = self.d
        
        # Compute global weight estimate and covariance
        try:
            A_inv_global = np.linalg.inv(self.A_global)
            w_hat = A_inv_global @ self.b_global
        except np.linalg.LinAlgError:
            w_hat = np.zeros(d)
            A_inv_global = np.eye(d)
        
        scores = []
        for a in range(self.n):
            # Mean reward estimate
            mean_global = w_hat @ context
            
            # Bias estimate
            try:
                A_inv_bias = np.linalg.inv(self.A_bias[a])
                b_hat = A_inv_bias @ self.b_bias[a]
            except np.linalg.LinAlgError:
                b_hat = np.array([0.0])
                A_inv_bias = np.array([[1.0]])
            
            mean_bias = b_hat[0] if b_hat.size > 0 else 0.0
            
            # Total mean
            mean = mean_global + mean_bias
            
            # Variance for exploration
            var_global = context @ A_inv_global @ context
            var_bias = A_inv_bias[0, 0] if A_inv_bias.size > 0 else 1.0
            total_var = var_global + var_bias
            
            # UCB score
            score = mean + self.alpha * np.sqrt(max(total_var, 0.0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the global model and arm-specific bias with new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Apply forgetting factor to maintain non-stationarity adaptation
        self.A_global = self.forgetting * self.A_global
        self.b_global = self.forgetting * self.b_global
        self.A_bias[arm] = self.forgetting * self.A_bias[arm]
        self.b_bias[arm] = self.forgetting * self.b_bias[arm]
        
        # Update global model with the centered reward (subtract current bias estimate)
        # First, get current bias estimate
        try:
            A_inv_bias = np.linalg.inv(self.A_bias[arm])
            b_hat = A_inv_bias @ self.b_bias[arm]
            bias_est = b_hat[0] if b_hat.size > 0 else 0.0
        except np.linalg.LinAlgError:
            bias_est = 0.0
        
        # Centered reward for global model
        centered_reward = reward - bias_est
        
        # Update global model
        self.A_global += np.outer(context, context)
        self.b_global += centered_reward * context
        
        # Update arm-specific bias
        # The bias model is: r - context^T w = b_a + noise
        # We approximate by using the residual after accounting for global model
        try:
            A_inv_global = np.linalg.inv(self.A_global)
            w_hat = A_inv_global @ self.b_global
            pred_global = w_hat @ context
        except np.linalg.LinAlgError:
            pred_global = 0.0
        
        residual = reward - pred_global
        self.A_bias[arm] += 1.0
        self.b_bias[arm] += residual
# EVOLVE-BLOCK-END
