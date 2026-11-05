"""Non-stationary linear contextual bandit with Sliding-Window RLS and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Sliding-Window RLS for per-arm
    weight estimation and Upper Confidence Bound (UCB) for exploration.
    
    This approach differs from simple EMA by tracking uncertainty (covariance)
    explicitly, allowing the exploration bonus to adapt to how much data has been
    seen recently for each arm, rather than using a fixed epsilon.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm RLS state:
        # P[a]: Inverse covariance matrix (d x d)
        # theta[a]: Weight vector (d x 1)
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]  # Large initial uncertainty
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # RLS parameters
        # Lambda controls the effective window size. 
        # Lambda close to 1 = long memory, Lambda smaller = short memory.
        # For non-stationary environments, we want a relatively short window.
        # 0.95 is a common choice, but we can go lower for faster adaptation.
        # Based on inspiration 1 and cell findings, aggressive forgetting helps.
        self.lam = 0.90
        
        # Ridge regularization for numerical stability
        self.ridge = 0.1

        # UCB exploration parameter
        # Higher alpha = more exploration
        self.alpha_ucb = 2.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

    def _check_deadline(self):
        """Return True if we should stop computing and use a safe fallback."""
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        """Select arm using UCB based on RLS estimates."""
        if self._check_deadline():
            # Fallback: pick arm with highest mean estimate (greedy)
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        scores = []
        for a in range(self.n):
            # Mean estimate
            mu = self.theta[a] @ context
            
            # Uncertainty bonus: sqrt(x^T P x)
            # P is the inverse covariance, so x^T P x is the variance of the estimate
            var = context @ self.P[a] @ context
            bonus = self.alpha_ucb * np.sqrt(var)
            
            scores.append(mu + bonus)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update RLS weights for the selected arm."""
        context = np.asarray(context, dtype=np.float64)
        
        # Standard RLS update with forgetting factor
        # P_new = (1/lam) * (P_old - (P_old x x^T P_old) / (lam + x^T P_old x))
        
        P = self.P[arm]
        x = context
        
        # Compute denominator
        xPx = x @ P @ x
        denom = self.lam + xPx
        
        # Update P
        # P_new = (1/lam) * P_old - (1/lam^2) * (P_old x x^T P_old) / (x^T P_old x / lam + 1)
        # Let's use the standard form:
        # P_new = (1/lambda) * (I - P x (lambda + x^T P x)^-1 x^T P)
        # Wait, standard RLS with forgetting factor lambda:
        # P_k = (1/lambda) P_{k-1} - (1/lambda^2) P_{k-1} x (x^T P_{k-1} x / lambda + 1)^-1 x^T P_{k-1}
        
        # Simpler and more stable form:
        # K = P x / (lambda + x^T P x)
        # P_new = (1/lambda) * (P - K x^T P)
        # theta_new = theta + K (y - x^T theta)
        
        K = P @ x / denom
        
        # Update P
        # Note: P is symmetric, so P - K x^T P = P - K x^T P
        # We can compute P_new = (1/lambda) * (P - np.outer(K, x @ P))
        # Since P is symmetric, x @ P = (P @ x)^T = K^T * denom
        # So x @ P = K.T * denom
        # P_new = (1/lambda) * (P - np.outer(K, K.T * denom))
        # P_new = (1/lambda) * (P - denom * np.outer(K, K.T))
        
        self.P[arm] = (1.0 / self.lam) * (P - xPx * np.outer(K, K))
        
        # Ensure symmetry (numerical drift can break it)
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        # Update theta
        error = reward - self.theta[arm] @ x
        self.theta[arm] = self.theta[arm] + K * error

# EVOLVE-BLOCK-END
