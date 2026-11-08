"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Thompson Sampling with adaptive forgetting for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Initialize matrices with small regularization to avoid singularity initially
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_forgetting = 0.995  # Default slow forgetting
        self.current_forgetting = self.base_forgetting
        
        # State tracking for adaptation detection
        self.last_rewards = [None] * n_arms
        self.reward_history = []
        
        # Time management for safety
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def _get_adaptive_forgetting(self, arm, reward):
        """
        Dynamically adjust the forgetting factor based on recent reward performance.
        If a reward is significantly lower than recent history for that arm,
        we increase the forgetting rate (decrease beta) to adapt faster.
        """
        if self.last_rewards[arm] is not None:
            # Simple heuristic: if reward drops by more than 20% relative to last, increase forgetting
            # Or if absolute drop is large. Let's use a relative drop check.
            if self.last_rewards[arm] > 0:
                drop_ratio = (self.last_rewards[arm] - reward) / self.last_rewards[arm]
                if drop_ratio > 0.1:
                    # Spike the forgetting: move closer to 1.0 (forget more)
                    # e.g., 0.995 -> 0.999
                    self.current_forgetting = min(0.999, self.current_forgetting * 1.01)
                else:
                    # Decay back to base
                    self.current_forgetting = max(self.base_forgetting, self.current_forgetting * 0.99)
            else:
                # If last reward was 0 or negative, just decay to base
                self.current_forgetting = max(self.base_forgetting, self.current_forgetting * 0.99)
        else:
            self.current_forgetting = self.base_forgetting

        self.last_rewards[arm] = reward

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy or cached best
        # However, the interface requires returning an arm. We don't have a global "done" state,
        # but we can ensure computation is fast.
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Compute posterior mean and variance
            # theta_hat = A^{-1} b
            # Var(theta) = A^{-1} / lambda (in standard TS, we often scale by 1)
            
            try:
                # Use solve for better numerical stability and speed than explicit inverse
                theta_hat = np.linalg.solve(self.A[a], self.b[a])
                cov = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular (shouldn't happen with eye init + updates)
                theta_hat = np.zeros(self.d)
                cov = np.eye(self.d)

            # Sample from N(theta_hat, cov)
            # np.random.multivariate_normal is slow. Use Cholesky decomposition for speed.
            try:
                L = np.linalg.cholesky(cov)
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                # If Cholesky fails (non-PD), add small jitter
                cov_jittered = cov + 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(cov_jittered)
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z

            scores.append(theta_sample @ context)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Update adaptive forgetting factor
        self._get_adaptive_forgetting(arm, reward)
        
        # Apply forgetting to existing statistics
        beta = self.current_forgetting
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context
        
        self.t += 1
# EVOLVE-BLOCK-END
