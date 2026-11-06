import numpy as np
import time
from collections import deque

class Policy:
    """
    Single Global Linear Model with Arm-Specific Biases for Non-Stationary Contextual Bandits.
    
    Strategy:
    - Instead of maintaining 5 separate linear models (one per arm), maintain ONE global linear model
      that captures the context-dependent trend, plus a scalar bias for each arm.
    - Reward model: r_t = w^T x_t + b_a + noise
    - This reduces the effective dimensionality of the learning problem from n_arms * dim to dim + n_arms.
    - In non-stationary settings, the global weights adapt to the common drift, while biases adjust for
      arm-specific offsets. This improves sample efficiency and generalization compared to per-arm models
      which may have insufficient data to track drift independently.
    
    Uses Recursive Least Squares (RLS) with exponential forgetting for the global model and
    simple exponential moving averages for the biases.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global Model: RLS state
        # P is the inverse covariance matrix (inverse of A)
        self.P = np.eye(dim) / 1.0  # Initial inverse covariance
        self.theta = np.zeros(dim)  # Global weights
        
        # Arm-specific biases
        self.biases = np.zeros(n_arms)
        # For bias tracking, we can use a simple EMA of the residuals attributed to bias
        # Or just track a running mean of (reward - w^T x) for each arm.
        # Let's use an exponential moving average for the bias estimate.
        self.bias_alpha = 0.1  # Learning rate for bias EMA
        
        # Forgetting factor for RLS
        self.lambda_rls = 0.95
        
        # Exploration
        self.alpha_ucb = 1.0
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0
        
        # History for potential adaptation (not strictly needed for core logic but good for safety)
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick arm with highest bias (if theta is zero, this is just max bias)
            # But we should still try to use theta if available
            scores = []
            for a in range(self.n):
                mu = self.theta @ context + self.biases[a]
                # No exploration bonus in fallback to save time, or simple constant
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Mean prediction
            mu = self.theta @ context + self.biases[a]
            
            # Uncertainty bonus
            # Variance of prediction from global model: x^T P x
            # Note: The bias uncertainty is not included in the UCB bonus here,
            # assuming bias is estimated separately. We could add a term, but keeping it simple.
            var = context @ (self.P @ context)
            bonus = self.alpha_ucb * np.sqrt(var)
            
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Update Global Model (RLS)
        # We need to remove the bias effect from the reward to update the global weights?
        # Or do we treat the bias as part of the noise?
        # If we treat bias as noise, the global model will try to fit the bias into weights,
        # which is bad if biases are large.
        # Better: Update global model using residual = reward - bias_estimate.
        # Then update bias estimate using the new residual from the global model.
        
        # Current bias estimate
        b_curr = self.biases[arm]
        
        # Residual for global model update
        pred_global = self.theta @ context
        residual_for_global = reward - b_curr
        
        # RLS Update for theta and P
        # P_new = (1/lambda) * P - (P x x^T P) / (lambda + x^T P x)
        # theta_new = theta + P_new x * (y - theta^T x)  ... wait, standard RLS formula:
        # K = P x / (lambda + x^T P x)
        # theta = theta + K * (y - theta^T x)
        # P = (1/lambda) * (P - K x^T P)  ... actually P = (1/lambda) * (P - P x x^T P / (lambda + x^T P x))
        
        Px = self.P @ context
        denom = self.lambda_rls + context @ Px
        
        if denom > 1e-10:
            # Gain
            K = Px / denom
            
            # Update theta
            error = residual_for_global - self.theta @ context
            self.theta = self.theta + K * error
            
            # Update P
            # P = (1/lambda) * P - (1/lambda) * K x^T P
            # Since K = Px / denom, K x^T P = (Px x^T P) / denom
            # P_new = (1/lambda) * (P - (Px x^T P) / denom)
            # Let's compute term = np.outer(Px, context @ self.P)
            # Note: context @ self.P is row vector x^T P
            # Px is column vector P x
            # So outer(Px, x^T P) is P x x^T P
            
            xP = context @ self.P  # shape (d,)
            term = np.outer(Px, xP) / denom
            self.P = (1.0 / self.lambda_rls) * (self.P - term)
        else:
            # Fallback: reset or ignore?
            pass

        # 2. Update Bias
        # Now that theta is updated, calculate the residual for the bias
        pred_new_global = self.theta @ context
        residual_for_bias = reward - pred_new_global
        
        # Update bias using EMA
        # self.biases[arm] = (1 - self.bias_alpha) * self.biases[arm] + self.bias_alpha * residual_for_bias
        # This is a simple, robust way to track the arm-specific offset.
        self.biases[arm] = (1.0 - self.bias_alpha) * self.biases[arm] + self.bias_alpha * residual_for_bias

        # 3. Optional: Adapt alpha_ucb
        # We could adapt alpha based on recent errors, but let's keep it fixed for now to avoid
        # introducing too many hyperparameters that might destabilize the new structure.
        # If we want, we can add a simple check.
        # For now, fixed alpha_ucb = 1.0 is a reasonable starting point for this structure.
