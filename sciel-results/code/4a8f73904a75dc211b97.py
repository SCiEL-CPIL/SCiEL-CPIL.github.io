"""Non-stationary linear contextual bandit policy with sliding window LinUCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary LinUCB with:
    - Sliding window of last W observations per arm
    - Exact removal of old observations from the Gram matrix
    - Fixed exploration parameter
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Sliding window size
        self.window_size = 50
        
        # Core matrices for LinUCB
        # We maintain A and b for the current window
        self.A = [np.zeros((dim, dim)) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Deques to store (context, reward) for each arm
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]

        # Exploration parameter
        self.alpha = 1.5

        # Regularization for numerical stability in solve
        self.reg = 1e-2

        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit

    def _is_time_up(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        self.t += 1

        # Deadline guard: if we're close to timeout, use greedy with best known arm
        if self._is_time_up():
            # Use the arm with highest recent average reward
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                if self.history[a]:
                    # Calculate mean of recent rewards in the deque
                    rewards = [r for _, r in self.history[a]]
                    mean_r = np.mean(rewards)
                    if mean_r > best_mean:
                        best_mean = mean_r
                        best_arm = a
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Add regularization to A for stability
            A_reg = self.A[a] + self.reg * np.eye(self.d)
            
            try:
                # Solve A @ theta = b
                theta = np.linalg.solve(A_reg, self.b[a])
                # Uncertainty: sqrt(x^T A^{-1} x)
                # Solve A @ y = x
                y = np.linalg.solve(A_reg, context)
                unc_sq = context @ y
                unc = np.sqrt(max(0.0, unc_sq))
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                unc = 0.0

            mean = theta @ context
            scores[a] = mean + self.alpha * unc

        # Clip scores to avoid extreme values
        scores = np.clip(scores, -10.0, 10.0)

        # Random tie-breaking
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # If window is full, remove the oldest observation
        if len(self.history[arm]) == self.window_size:
            old_context, old_reward = self.history[arm].popleft()
            # Subtract the contribution of the old observation
            self.A[arm] -= np.outer(old_context, old_context)
            self.b[arm] -= old_reward * old_context

        # Add the new observation
        self.history[arm].append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Ensure numerical stability: if A becomes ill-conditioned or negative due to floating point errors,
        # we might need to renormalize or reset. 
        # However, with a small window and double precision, this is usually stable.
        # We can check if any eigenvalue is negative (which shouldn't happen for PSD matrices)
        # but doing eigendecomposition every step is expensive.
        # Instead, we just rely on the regularization in select().
# EVOLVE-BLOCK-END
