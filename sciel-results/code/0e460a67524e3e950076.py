"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with adaptive covariance inflation for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Per-arm posterior parameters: A (precision) and b (cross-moment)
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive inflation base and current multiplier
        self.base_inflation = 1.005
        self.inflation = 1.005
        # Track reward statistics for adaptive inflation
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit
        self.elapsed = 0.0

    def select(self, context):
        # Deadline guard
        self.elapsed = time.time() - self.start_time
        if self.elapsed > self.deadline:
            # Return greedy arm based on current estimates
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        # Sample from posterior for each arm
        scores = []
        for a in range(self.n):
            # Compute posterior mean and covariance
            inv_A = np.linalg.inv(self.A[a])
            theta_mean = inv_A @ self.b[a]
            theta_std = np.sqrt(np.diag(inv_A))
            
            # Sample theta from N(theta_mean, inv_A)
            # Use Cholesky for sampling
            try:
                L = np.linalg.cholesky(inv_A)
            except np.linalg.LinAlgError:
                L = np.eye(self.d)
            
            z = np.random.randn(self.d)
            theta_sample = theta_mean + L @ z
            
            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        self.elapsed = time.time() - self.start_time
        if self.elapsed > self.deadline:
            return

        # Update reward statistics for adaptive inflation
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_var += (delta * (reward - self.reward_mean) - self.reward_var) / max(self.reward_count, 1)
        self.reward_var = max(self.reward_var, 1e-6)

        # Compute standardized innovation magnitude
        # Estimate prediction variance for this arm
        inv_A = np.linalg.inv(self.A[arm])
        pred_var = context @ inv_A @ context
        pred_std = np.sqrt(max(pred_var, 1e-6))
        
        # Standardized innovation
        innovation = abs(reward - (inv_A @ self.b[arm]) @ context) / pred_std if pred_std > 1e-6 else 0.0

        # Adaptive inflation: increase when innovation is large
        # Use a smooth function to modulate inflation
        if innovation > 2.0:
            self.inflation = min(1.05, self.base_inflation + 0.02 * (innovation - 2.0))
        elif innovation < 1.0:
            self.inflation = max(self.base_inflation * 0.95, self.inflation * 0.99)
        else:
            self.inflation = self.base_inflation

        # Update with adaptive inflation (forgetting factor)
        self.A[arm] = self.inflation * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.inflation * self.b[arm] + reward * context

        # Occasionally re-normalize to prevent numerical issues
        if self.reward_count % 100 == 0:
            for a in range(self.n):
                # Ensure A remains positive definite
                eigvals = np.linalg.eigvalsh(self.A[a])
                if np.min(eigvals) < 1e-4:
                    self.A[a] += 1e-3 * np.eye(self.d)
# EVOLVE-BLOCK-END
