# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with per-arm exponentially weighted
    ridge regression and adaptive UCB exploration.
    
    Key structural choices:
    - Per-arm linear models (standard for linear bandits)
    - Exponential forgetting (gamma) to handle non-stationarity
    - Adaptive beta that increases exploration when uncertainty is high
    - Efficient incremental matrix updates with Cholesky or direct inverse
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.99  # forgetting factor
        self.ridge = 1e-2  # regularization
        self.beta_base = 1.0  # base exploration parameter
        self.t = 0  # time step counter
        
        # Per-arm state
        self.A = [self.ridge * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.ridge for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # seconds (10s safety margin before 150s)
        
        # Track best arm for fallback if time runs out
        self.best_arm = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: if close to deadline, return best known arm
        if time.time() > self.deadline:
            return self.best_arm
        
        self.t += 1
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mu = theta @ context
            
            # Uncertainty bonus
            xAx = context @ self.A_inv[a] @ context
            # Adaptive beta: scale with sqrt of time since last update or just use base
            beta = self.beta_base * np.sqrt(1.0 + 0.1 * self.t * 0.01)
            bonus = beta * np.sqrt(max(0.0, xAx))
            
            scores[a] = mu + bonus
        
        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline guard
        if time.time() > self.deadline:
            return
        
        # Update per-arm model with exponential forgetting
        g = self.gamma
        self.A[arm] = g * self.A[arm] + np.outer(context, context)
        self.b[arm] = g * self.b[arm] + reward * context
        
        # Update inverse using Sherman-Morrison or direct recompute
        # For efficiency, use direct inverse computation (dim is small)
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Fallback: add more regularization
            self.A[arm] += 1e-6 * np.eye(self.d)
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d) / self.ridge
# EVOLVE-BLOCK-END
