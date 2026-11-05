import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # RLS (Recursive Least Squares) parameters
        # Lambda (forgetting factor): 0.95 is a standard choice for non-stationary environments.
        # It effectively gives weight to the last ~20 samples exponentially.
        self.lam = 0.95
        
        # Regularization to prevent singularity at start
        self.reg = 1.0
        
        # Per-arm state for RLS
        # P: Inverse covariance matrix (d x d)
        # theta: Weight vector (d)
        # We initialize P as (1/reg) * I and theta as 0.
        # This corresponds to a strong prior that weights are 0.
        
        self.P = [np.eye(self.d) / self.reg for _ in range(self.n)]
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        
        # UCB parameter
        # alpha controls the exploration-exploitation tradeoff.
        # Higher alpha means more exploration.
        # In non-stationary settings, we might want a slightly higher alpha
        # to keep exploring as the environment changes.
        self.alpha = 1.0
        
        # Time step counter
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Predicted mean
            mu = np.dot(self.theta[a], context_vec)
            
            # Uncertainty term for UCB
            # Var = x^T P x
            # We use the diagonal of P x x^T P? No, just x^T P x.
            # P is the inverse covariance of the estimator.
            # The variance of the prediction is x^T P x.
            # We take the square root to get the standard deviation.
            px = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, px)
            
            # Ensure variance is non-negative (numerical stability)
            if var < 0:
                var = 0.0
                
            std = np.sqrt(var)
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # RLS Update for the selected arm
        # 1. Calculate gain vector K
        # K = P x / (lam + x^T P x)
        
        P_x = np.dot(self.P[arm], context_vec)
        denom = self.lam + np.dot(context_vec, P_x)
        
        # Avoid division by zero
        if denom < 1e-12:
            denom = 1e-12
            
        K = P_x / denom
        
        # 2. Update theta
        # Error = y - x^T theta
        pred = np.dot(self.theta[arm], context_vec)
        error = reward - pred
        
        self.theta[arm] += K * error
        
        # 3. Update P
        # P_new = (1/lam) * (P - K x^T P)
        # Note: The standard RLS update for inverse covariance with forgetting factor lambda is:
        # P(t) = 1/lambda * (P(t-1) - K(t) x(t) x(t)^T P(t-1))
        # Or equivalently: P(t) = 1/lambda * (P(t-1) - K(t) P_x^T) where P_x = P(t-1) x(t)
        
        self.P[arm] = (1.0 / self.lam) * (self.P[arm] - np.outer(K, P_x))
        
        # Numerical stability: Symmetrize P
        # Due to floating point errors, P might become slightly asymmetric.
        # P = (P + P.T) / 2
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        self.t += 1
