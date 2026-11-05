"""Non-stationary linear contextual bandit with adaptive Kalman filter and residual-driven process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Per-Arm Robust Recursive Least Squares (RLS).
    
    Design:
    - Each arm has its own weight vector theta_a and inverse covariance matrix P_a.
    - RLS update with a forgetting factor (lambda) to handle non-stationarity.
    - Exploration bonus based on the prediction variance (context^T P context).
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Per-arm parameters
        # theta: list of (dim,) arrays
        # P: list of (dim, dim) matrices
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        # Initialize P with large value to encourage exploration
        self.P = [np.eye(self.d) * 10.0 for _ in range(self.n)]
        
        # RLS parameters
        self.forgetting_factor = 0.99  # Controls adaptation speed
        self.alpha = 2.0  # Exploration coefficient
        self.alpha_decay = 0.9995
        self.alpha_min = 0.5
        
        # Tracking
        self.round_num = 0
        self.arm_counts = [0] * self.n

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # Variance: context^T P_a context
            # Use diagonal approximation for speed if dim is large, but exact is fine for dim=5
            var = context @ self.P[a] @ context
            # Ensure variance is non-negative
            var = max(var, 0.0)
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        # Decay alpha slightly to reduce exploration over time
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        # RLS Update for the selected arm
        # Prediction
        y_hat = self.theta[arm] @ context
        innovation = reward - y_hat
        
        # Gain calculation
        # K = P x / (lambda + x^T P x)
        P = self.P[arm]
        Px = P @ context
        denom = self.forgetting_factor + context @ Px
        
        if denom < 1e-10:
            # Avoid division by zero
            K = np.zeros(self.d)
        else:
            K = Px / denom
            
        # Update theta
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Update P (Joseph form or standard RLS form)
        # P_new = (1/lambda) * (P - K x^T P)
        # Or: P_new = (1/lambda) * P - (1/lambda^2) * (P x x^T P) / (1/lambda + x^T P x)
        # Standard efficient form:
        self.P[arm] = (P - np.outer(K, context @ P)) / self.forgetting_factor
        
        # Symmetrize to prevent numerical drift
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)
        
        # Ensure positive semi-definite (clip negative eigenvalues if any, though RLS should maintain PSD)
        # For small dims, we can just check diagonal
        np.fill_diagonal(self.P[arm], np.maximum(np.diag(self.P[arm]), 1e-6))

        self.arm_counts[arm] += 1
# EVOLVE-BLOCK-END
