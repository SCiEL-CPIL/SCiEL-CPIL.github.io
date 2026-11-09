import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive Sliding Window Linear Thompson Sampling for Non-Stationary Linear Bandits.
    
    Structural Design:
    1. Per-arm sliding window model (from Inspiration 1's robust structure) to handle non-stationarity.
    2. Adaptive window size per arm based on EWMA prediction error (from Inspiration 2's refinement).
       - High error -> shrink window (faster adaptation).
       - Low error -> grow window (lower variance).
    3. Thompson Sampling with Cholesky-based sampling for efficient exploration.
    4. Deadline guard to ensure termination.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit
        
        # Initial window size
        self.base_window_size = 40
        self.min_window_size = 10
        self.max_window_size = 150
        
        # Ridge regularization parameter
        self.lam = 1.0
        
        # Per-arm sliding window data
        # queues[a] stores (context, reward) in chronological order
        # We don't use maxlen in deque to allow dynamic resizing of the effective window
        # instead we manage the length manually to adapt size.
        self.queues = [deque() for _ in range(n_arms)]
        
        # Cached A and b matrices for fast updates
        # A[a] = sum of x_i x_i^T over window, b[a] = sum of r_i x_i
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA tracking for non-stationarity detection
        # ewma_sq_err[a] tracks exponentially weighted mean squared prediction error
        self.ewma_sq_err = [0.0] * n_arms
        self.ewma_alpha = 0.3  # Fast response to changes
        
        # Effective window size per arm (adapted based on non-stationarity)
        self.eff_window = [self.base_window_size] * n_arms
        
        # Precompute Cholesky factorizations lazily
        self._chol_cache = [None] * n_arms

    def _update_arm_stats(self, arm, context, reward):
        """Update the sliding window statistics for a given arm."""
        q = self.queues[arm]
        
        # If the queue is larger than the current effective window, we need to remove oldest
        # Note: We might have shrunk the window, so we need to trim down to eff_window
        # Or if we just added and it exceeds eff_window, trim.
        
        # First, add the new observation
        q.append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Now, if the queue is longer than the effective window, remove oldest
        # This handles both the case where window size is fixed and we exceed it,
        # and the case where window size has shrunk.
        while len(q) > self.eff_window[arm]:
            old_ctx, old_reward = q.popleft()
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_reward * old_ctx
            
        # Invalidate Cholesky cache
        self._chol_cache[arm] = None

    def _adapt_window_size(self, arm, context, reward):
        """Adapt the effective window size based on prediction error."""
        # Compute current prediction
        A = self.A[arm]
        b = self.b[arm]
        
        try:
            theta = np.linalg.solve(A, b)
            pred = theta @ context
            sq_err = (reward - pred) ** 2
        except np.linalg.LinAlgError:
            sq_err = 0.0
            
        # Update EWMA
        self.ewma_sq_err[arm] = (1 - self.ewma_alpha) * self.ewma_sq_err[arm] + self.ewma_alpha * sq_err
        
        # Adjust window size
        # If error is high, shrink window to forget old data faster
        # If error is low, grow window to reduce variance
        if self.ewma_sq_err[arm] > 1.0:  # Threshold for high error
            self.eff_window[arm] = max(self.min_window_size, int(self.eff_window[arm] * 0.9))
        elif self.ewma_sq_err[arm] < 0.5:  # Threshold for low error
            self.eff_window[arm] = min(self.max_window_size, int(self.eff_window[arm] * 1.1))
            
        # Ensure window size is at least 1
        self.eff_window[arm] = max(1, self.eff_window[arm])

    def select(self, context):
        """
        Select an arm using Thompson Sampling.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check: if we're running out of time, use a simple greedy fallback
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    pass
            return int(best_arm)
        
        samples = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Compute posterior mean
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                
            # Sample from N(theta, A^{-1})
            # Use Cholesky for efficiency and stability
            try:
                # Add small jitter for numerical stability
                A_reg = A + 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(A_reg)
                z = np.random.randn(self.d)
                w = np.linalg.solve(L.T, z)
                # Covariance is A^{-1}, so sample is theta + A^{-1/2} z
                # Note: A^{-1/2} z = L^{-T} L^{-1} z? No.
                # If A = L L^T, then A^{-1} = L^{-T} L^{-1}.
                # We want x ~ N(theta, A^{-1}).
                # x = theta + A^{-1/2} z.
                # A^{-1/2} = L^{-T} L^{-1} is not right.
                # A^{-1/2} = (L L^T)^{-1/2} = L^{-T} (L^{-T} L^{-1})^{1/2}?
                # Standard way: A^{-1/2} z = L^{-T} u where u ~ N(0, I)?
                # Let's verify: Var(L^{-T} u) = L^{-T} I L^{-1} = (L L^T)^{-1} = A^{-1}. Correct.
                # So we need to solve L^T y = z, then x = theta + L^{-T} z?
                # Wait. L is lower triangular. A = L L^T.
                # We want v such that v v^T = A^{-1}.
                # v = L^{-T}.
                # So sample = theta + L^{-T} z.
                # To compute L^{-T} z, we solve L^T y = z.
                w = np.linalg.solve(L.T, z)
                theta_sample = theta + w
            except np.linalg.LinAlgError:
                # Fallback to diagonal approximation or just mean
                try:
                    # Use inverse if Cholesky fails
                    cov = np.linalg.inv(A)
                    theta_sample = theta + np.random.multivariate_normal(np.zeros(self.d), cov)
                except:
                    theta_sample = theta
                    
            pred_reward = theta_sample @ context
            samples.append(pred_reward)
            
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """
        Update the policy with a new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Adapt window size first, so we know how much to trim
        self._adapt_window_size(arm, context, reward)
        
        # Update statistics
        self._update_arm_stats(arm, context, reward)
