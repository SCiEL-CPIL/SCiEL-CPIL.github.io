"""
Non-stationary linear contextual bandit policy.
Uses Sliding Window Linear Bandit with Thompson Sampling.
Structural change: Replaces exponential forgetting with a hard sliding window
and UCB with Thompson Sampling for better exploration in non-stationary settings.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_SAFE = 135.0

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.feature_dim = dim
        self.start_time = time.time()
        
        # Sliding Window parameters
        # Hard cutoff on old data allows faster adaptation to abrupt changes
        self.window_size = 50 
        self.max_samples = 200 # Cap memory for numerical stability and speed
        
        # Regularization for stability
        self.lambda_reg = 1.0
        
        # Per-arm data storage (circular buffers or lists)
        # We store (context, reward) pairs
        self.data = [[] for _ in range(n_arms)]
        
        # Pre-allocated matrices for fast updates (approximate)
        # We will recompute A and b from the window periodically or incrementally
        # For efficiency, we maintain A and b incrementally but prune old data
        # Note: Maintaining exact A/b with a sliding window is O(N) per update if we remove old items.
        # Given small window (50), O(N) is fast enough.
        # A = sum(x x^T), b = sum(r x)
        self.A = [np.eye(self.feature_dim) * self.lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(self.feature_dim) for _ in range(n_arms)]
        
        # Store history to allow removal
        # history[a] is a list of (context, reward)
        
        # Thompson Sampling parameters
        self._rng = np.random.RandomState(42)
        
        self.t = 0

    def _get_context(self, context):
        """Ensure context is a 1D numpy array."""
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            return context
        return context.reshape(-1)

    def _recompute_arm(self, arm):
        """Recompute A and b from the sliding window for a specific arm."""
        # Clear and recompute
        self.A[arm] = np.eye(self.feature_dim) * self.lambda_reg
        self.b[arm] = np.zeros(self.feature_dim)
        
        data = self.data[arm]
        if not data:
            return
            
        # Vectorized update
        # data is list of (x, r)
        # Build matrix X (N x D) and vector r (N)
        N = len(data)
        X = np.zeros((N, self.feature_dim))
        r = np.zeros(N)
        for i, (x, ri) in enumerate(data):
            X[i] = x
            r[i] = ri
            
        self.A[arm] += X.T @ X
        self.b[arm] += X.T @ r

    def select(self, context):
        """Select arm using Thompson Sampling."""
        context = self._get_context(context)
        
        # Deadline guard
        if time.time() - self.start_time > _SAFE:
            # Fallback to greedy based on current estimates
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n_arms):
                try:
                    # Solve (A + lambda I) theta = b
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    val = float(np.dot(theta, context))
                    if val > best_val:
                        best_val = val
                        best_arm = a
                except:
                    val = 0.0
                    if val > best_val:
                        best_val = val
                        best_arm = a
            return int(best_arm)

        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            try:
                # Compute mean estimate
                # A is (D x D), b is (D)
                # theta = A^{-1} b
                # We use solve for stability
                theta_mean = np.linalg.solve(self.A[a], self.b[a])
                
                # Compute covariance for Thompson Sampling
                # Cov = A^{-1}
                # We need to sample theta ~ N(theta_mean, Cov)
                
                # Use Cholesky for sampling
                L = np.linalg.cholesky(self.A[a])
                
                # Sample z ~ N(0, I)
                z = self._rng.randn(self.feature_dim)
                
                # Solve L v = z => v = L^{-1} z
                v = np.linalg.solve(L, z)
                
                # theta_sample = theta_mean + A^{-1/2} z = theta_mean + L^{-T} v
                # Since A = L L^T, A^{-1} = L^{-T} L^{-1}
                # Cov^{1/2} z = L^{-T} (L^{-1} z)
                # So theta_sample = theta_mean + L^{-T} v
                theta_sample = theta_mean + np.linalg.solve(L.T, v)
                
                score = float(np.dot(theta_sample, context))
                
                if score > best_score:
                    best_score = score
                    best_arm = a
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter and retry or fallback to random
                try:
                    self.A[a] += 1e-4 * np.eye(self.feature_dim)
                    L = np.linalg.cholesky(self.A[a])
                    theta_mean = np.linalg.solve(self.A[a], self.b[a])
                    z = self._rng.randn(self.feature_dim)
                    v = np.linalg.solve(L, z)
                    theta_sample = theta_mean + np.linalg.solve(L.T, v)
                    score = float(np.dot(theta_sample, context))
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    score = self._rng.randn() * 10.0
                    if score > best_score:
                        best_score = score
                        best_arm = a

        return int(best_arm)

    def update(self, context, arm, reward):
        """Update the per-arm model with sliding window linear regression."""
        context = self._get_context(context)
        a = int(arm)
        r = float(reward)
        
        # Add new data point
        self.data[a].append((context, r))
        
        # Check if we need to prune
        if len(self.data[a]) > self.max_samples:
            # Remove oldest
            self.data[a].pop(0)
            
            # For efficiency, we can update A and b incrementally by subtracting the removed point
            # However, floating point errors might accumulate.
            # Given small window, recompute is safer and fast enough for D < 100.
            # Let's recompute if we pruned, or just recompute every K steps?
            # Recomputing every update is O(N*D^2). N=50, D=5 -> 1250 ops. Very fast.
            self._recompute_arm(a)
        else:
            # Incremental update is faster if no pruning
            self.A[a] += np.outer(context, context)
            self.b[a] += r * context
            
        self.t += 1

# EVOLVE-BLOCK-END
