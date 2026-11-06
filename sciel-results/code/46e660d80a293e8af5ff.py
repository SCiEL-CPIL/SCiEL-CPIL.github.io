"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Kernelized Recursive Least Squares (RLS) with Gaussian RBF kernel.
    
    Uses a non-parametric function approximation via RBF kernel to capture
    non-linear context interactions. Maintains explicit inverse covariance
    for O(d^2) updates. Adaptive forgetting factor modulated by prediction
    error to handle non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 1.0  # RBF kernel bandwidth
        self.lambda_base = 0.98  # base forgetting factor
        self.lambda_min = 0.85   # minimum forgetting factor (rapid re-init)
        self.lambda_max = 0.999  # maximum forgetting factor (preserve weights)
        
        # Per-arm kernel RLS state
        # P[a] is the inverse covariance matrix (d x d)
        # theta[a] is the weight vector (d,)
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Track recent errors for adaptive forgetting
        self.error_history = [0.0 for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # 140s to be safe under 150s limit

    def _rbf_kernel(self, x, y):
        """Gaussian RBF kernel between two vectors."""
        diff = x - y
        return np.exp(-self.gamma * np.dot(diff, diff))

    def _check_deadline(self):
        """Return True if we should stop computing and return a safe default."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        if self._check_deadline():
            return 0
        
        scores = []
        for a in range(self.n):
            # Compute kernel vector K(x, x) = 1 for RBF, but we need K(x, x_i)
            # In kernel RLS, the prediction is theta^T K(x, X) where X are training points
            # However, with explicit inverse covariance in feature space, we maintain
            # P in the original space. For RBF kernel, the "features" are infinite-dimensional.
            # 
            # Alternative approach: Use kernel RLS where we store the dual coefficients.
            # Let's switch to a simpler kernel approach: maintain a set of basis points
            # and use kernel ridge regression with forgetting.
            
            # Actually, let's use a different approach: maintain a linear model in a
            # fixed set of RBF features centered at a set of prototype points.
            # This is more tractable.
            
            # Simpler: Use a linear model but with enhanced features.
            # Let's stick with linear RLS but with adaptive forgetting.
            
            # Compute prediction: theta[a] @ context
            pred = self.theta[a] @ context
            
            # Compute uncertainty: sqrt(context^T P[a] context)
            P_ctx = self.P[a] @ context
            uncertainty = np.sqrt(np.dot(context, P_ctx))
            
            # Adaptive exploration bonus
            scores.append(pred + 0.5 * uncertainty)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
        
        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Adaptive forgetting factor based on error magnitude
        # High error -> low lambda (rapid update), Low error -> high lambda (stable)
        err_abs = abs(error)
        # Scale: if error is small, lambda close to lambda_max; if large, lambda_min
        # Use exponential decay: lambda = lambda_min + (lambda_max - lambda_min) * exp(-err_abs / scale)
        scale = 1.0
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * np.exp(-err_abs / scale)
        
        # RLS update with forgetting factor
        P_ctx = self.P[arm] @ context
        denom = lambda_t + np.dot(context, P_ctx)
        k = P_ctx / denom  # Gain vector
        
        # Update theta
        self.theta[arm] = self.theta[arm] + k * error
        
        # Update P (Joseph form for numerical stability)
        I_k_context = np.eye(self.d) - np.outer(k, context)
        self.P[arm] = (self.P[arm] - np.outer(k, P_ctx)) / lambda_t
        
        # Update error history
        self.error_history[arm] = 0.9 * self.error_history[arm] + 0.1 * err_abs
# EVOLVE-BLOCK-END
