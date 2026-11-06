"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Sliding Window LinUCB for non-stationary linear bandits.
    
    Uses a hard sliding window of the most recent W observations per arm.
    This completely discards old data, which is advantageous in non-stationary
    environments where the underlying parameters change over time.
    
    Efficiency:
    - Maintains A (covariance) and b (moment) incrementally.
    - Recomputes inverse and theta on every update (O(d^3) per arm).
    - This is feasible for small d (e.g., < 20) and fast enough for the time limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Window size for sliding window regression
        # A smaller window adapts faster but has higher variance.
        # A larger window is more stable but slower to adapt.
        self.window_size = 50
        
        # Regularization parameter
        self.beta = 1.0
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Per-arm storage
        self.A = [np.eye(dim) * self.beta for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) / self.beta for _ in range(n_arms)]
        
        # Deques to store (context, reward) pairs for the sliding window
        self.windows = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

    def _elapsed(self):
        return time.time() - self._start_time

    def _update_arm_model(self, arm):
        """
        Recalculate inv_A and theta for the given arm based on current A and b.
        """
        # A is already maintained incrementally. 
        # We need inv(A). 
        # Adding beta*I is already done in init, but we must ensure A is positive definite.
        # Since we start with beta*I and add xx^T, it is PD.
        # When we remove xx^T, it might lose PD-ness if the window is small or samples are collinear.
        # To be safe, we add a small epsilon or rely on the initial beta.
        # Actually, if we remove a sample that was crucial for PD-ness, A might become singular.
        # Let's add a small regularization on the fly if needed, or just try/except.
        
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.inv_A[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            # Fallback: add a small ridge
            self.inv_A[arm] = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.d))
            self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def select(self, context):
        # Deadline check
        if self._elapsed() > self._deadline:
            # Fallback: greedy
            scores = []
            for a in range(self.n):
                scores.append(float(self.theta[a] @ context))
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n)
        
        for a in range(self.n):
            # Mean prediction
            mu = float(self.theta[a] @ context)
            
            # Variance term
            var = float(context @ self.inv_A[a] @ context)
            std = np.sqrt(max(var, 0.0))
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline check
        if self._elapsed() > self._deadline:
            return

        # 1. Remove oldest sample if window is full
        if len(self.windows[arm]) == self.window_size:
            old_x, old_r = self.windows[arm][0]
            # Subtract contribution of old sample
            self.A[arm] -= np.outer(old_x, old_x)
            self.b[arm] -= old_r * old_x
            
        # 2. Add new sample
        self.windows[arm].append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # 3. Update model (inverse and theta)
        self._update_arm_model(arm)

# EVOLVE-BLOCK-END
