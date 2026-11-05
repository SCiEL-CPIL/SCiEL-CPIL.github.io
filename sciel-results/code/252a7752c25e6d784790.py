# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-Arm Non-Stationary Linear Bandit with Adaptive Regularized Least Squares (ARLS).
    
    Uses a recursive sliding-window update with adaptive regularization.
    The regularization strength is dynamically scaled based on recent innovation variance
    relative to context norm, allowing the policy to adapt its effective memory horizon.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0  # Safety margin before 120s limit

        # Per-arm models: A (matrix) and b (vector)
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]  # Small ridge init
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting factor per arm
        self.forging_factors = [0.99 for _ in range(n_arms)]

        # Track recent rewards and innovations for adaptive regularization
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.recent_contexts = [[] for _ in range(n_arms)]
        self.window_size = 20

        # Exploration parameter
        self.alpha = 1.0

        # Cache for inverse matrices (recomputed periodically)
        self.A_inv = [np.linalg.inv(self.A[i]) for i in range(n_arms)]

    def _compute_adaptive_forgetting(self, arm):
        """Compute adaptive forgetting factor based on recent innovation variance."""
        rewards = self.recent_rewards[arm]
        contexts = self.recent_contexts[arm]
        
        if len(rewards) < 5:
            return 0.99
        
        # Compute recent reward variance
        recent_r = np.array(rewards[-self.window_size:])
        recent_c = np.array(contexts[-self.window_size:])
        
        reward_var = np.var(recent_r)
        
        # Compute context norm
        context_norms = np.linalg.norm(recent_c, axis=1)
        avg_context_norm = np.mean(context_norms) if len(context_norms) > 0 else 1.0
        
        # Scale forgetting factor: higher variance -> more forgetting (lower factor)
        # Normalize variance to [0, 1] range approximately
        norm_factor = min(1.0, reward_var / (1.0 + avg_context_norm * 0.1))
        
        # Forgetting factor between 0.95 (high variance) and 0.999 (low variance)
        gamma = 0.999 - 0.049 * norm_factor
        gamma = np.clip(gamma, 0.95, 0.999)
        
        return gamma

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            context = np.array(context)
            scores = []
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        context = np.array(context)
        scores = []
        
        for a in range(self.n):
            # Compute theta = A^{-1} b
            theta = self.A_inv[a] @ self.b[a]
            
            # Prediction
            pred = theta @ context
            
            # Uncertainty estimate
            try:
                var = context @ self.A_inv[a] @ context
                if var < 0:
                    var = 0
            except:
                var = 0.0
            
            # UCB score with adaptive exploration
            bonus = self.alpha * np.sqrt(var)
            scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.array(context)

        # Update recent reward and context history
        self.recent_rewards[arm].append(reward)
        self.recent_contexts[arm].append(context)
        if len(self.recent_rewards[arm]) > self.window_size:
            self.recent_rewards[arm].pop(0)
            self.recent_contexts[arm].pop(0)

        # Compute adaptive forgetting factor
        gamma = self._compute_adaptive_forgetting(arm)
        self.forging_factors[arm] = gamma

        # Recursive sliding-window update
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodically recompute inverse for numerical stability
        if self.t % 10 == 0 or self.t == 1:
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                # Fallback: add ridge regularization
                self.A[arm] = self.A[arm] + 0.1 * np.eye(self.d)
                try:
                    self.A_inv[arm] = np.linalg.inv(self.A[arm])
                except:
                    self.A_inv[arm] = np.eye(self.d)
        # EVOLVE-BLOCK-END
