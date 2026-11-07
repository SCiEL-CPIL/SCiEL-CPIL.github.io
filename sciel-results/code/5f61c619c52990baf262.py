"""Rewritten program using GP-UCB (Model-and-Optimize) approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate and LCB acquisition.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_min = -5.0
        self.x_max = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        # 1. Initialization: Sample a few points
        # Use a small Latin Hypercube or just random for simplicity
        rng = np.random.default_rng(42)
        
        # Initial points: 10 random
        n_init = 10
        X_init = rng.uniform(self.x_min, self.x_max, size=(n_init, self.dim))
        Y_init = np.array([func(x) for x in X_init])
        
        self.X = X_init
        self.Y = Y_init
        
        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()
        
        calls_made = n_init
        
        # GP Parameters
        # Length scale
        l = 0.8 
        # Signal variance
        sigma_f_sq = 1.0
        # Noise
        sigma_n_sq = 1e-6
        
        # Acquisition: LCB (Lower Confidence Bound) for minimization
        # Minimize: mu(x) - beta * sigma(x)
        # We want to find the point that minimizes this value.
        # Beta is the exploration-exploitation trade-off.
        # A fixed beta or an adaptive beta can be used.
        # Let's use a fixed beta that balances exploration.
        beta = 2.0
        
        # Polish phase parameters
        polish_start_ratio = 0.8
        polish_calls_reserved = int(self.budget * (1 - polish_start_ratio))
        polish_threshold = 1e-3 # If improvement is small, consider polishing
        
        # Pre-compute kernel matrix and inverse for GP
        # Note: Refitting GP from scratch every step is O(N^3). 
        # With N=500, N^3 is 125M ops. Might be slow in Python.
        # Optimizing the acquisition function requires evaluating the GP at many points.
        # Evaluating GP at M points is O(M * N * D + M^2) if not careful, or O(M * (N+1) * D) for vectorized ops.
        # Fitting (inverting N x N matrix) is the bottleneck.
        # 500 steps * 500^3 is too slow. 
        # We need to either:
        # a) Use a cheaper surrogate (e.g., Kriging with fixed parameters, or just RBF interpolation).
        # b) Update the inverse incrementally (Sherman-Morrison).
        # c) Use a simpler model like Random Forest (but sklearn not guaranteed).
        # d) Limit the number of GP iterations or use a coarser model.
        
        # Actually, 500^3 = 125,000,000. In NumPy, matrix inversion of 500x500 is very fast (ms).
        # 500 inversions * 5ms = 2.5 seconds. This is well within the 300s limit.
        # The bottleneck will be evaluating the acquisition function.
        # We sample 2000 candidates per step.
        # Evaluating GP mean and variance at 2000 points:
        # K(x_new, X_old) is 2000x500.
        # Vectorized: O(2000 * 500 * 5) = 5M ops.
        # 500 steps * 5M = 2.5B ops. This might be ~1-5 seconds in NumPy.
        # Total time estimate: < 10-20 seconds. Safe.

        def compute_kernel(X1, X2):
            # RBF kernel
            # X1: (N1, D), X2: (N2, D)
            # Returns (N1, N2)
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            # Clip for numerical stability
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-sq_dist / (2 * l**2))

        # Determine when to switch to polishing
        polish_start_call = self.budget - polish_calls_reserved
        
        while calls_made < self.budget:
            if time.time() > deadline - 10:
                break
            
            # Check if we should switch to local polish
            if calls_made >= polish_start_call:
                # Use remaining budget for local optimization (Nelder-Mead)
                # We will use the remaining calls to polish best_x
                remaining_calls = self.budget - calls_made
                if remaining_calls > 0:
                    # Simple local search: Nelder-Mead
                    # We need to be careful with the budget
                    # Nelder-Mead takes ~dim * iterations calls
                    # Let's use a simple gradient-free local search or just perturbations
                    # Since we have a budget, we can do a few steps of Nelder-Mead
                    # But implementing Nelder-Mead from scratch is risky.
                    # Let's use a simpler "polish" strategy: sample points in a small neighborhood of best_x
                    # and pick the best.
                    radius = 0.5
                    # Shrink radius over time
                    for _ in range(remaining_calls):
                        if time.time() > deadline - 5:
                            break
                        # Sample in a box around best_x
                        x_polish = best_x + rng.uniform(-radius, radius, size=self.dim)
                        # Clip to bounds
                        x_polish = np.clip(x_polish, self.x_min, self.x_max)
                        y_polish = func(x_polish)
                        calls_made += 1
                        if y_polish < best_y:
                            best_y = y_polish
                            best_x = x_polish
                        # Shrink radius slightly
                        radius *= 0.9
                    break # Exit the main GP loop
                
            # Fit GP:
            # K(X, X) + sigma_n^2 I
            K = compute_kernel(self.X, self.X)
            K += (sigma_n_sq + 1e-8) * np.eye(len(self.X))
            
            # Standardize Y? 
            # It helps with numerical stability and scale of beta.
            y_mean = self.Y.mean()
            y_std = self.Y.std()
            if y_std < 1e-8: y_std = 1.0
            Y_std = (self.Y - y_mean) / y_std
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                # Re-use L from cholesky
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                # Fallback to inv
                alpha = np.linalg.inv(K) @ Y_std
                # If cholesky failed, we can't use L for variance, need to recompute or handle
                # For simplicity, if cholesky fails, we might skip this step or use a fallback
                # But let's assume it's rare.
                # We need L for the variance calculation below.
                # If L is not available, we can't compute variance efficiently.
                # Let's add a small jitter and retry cholesky.
                K += 1e-6 * np.eye(len(self.X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
                except np.linalg.LinAlgError:
                    # If it still fails, break or use a random point
                    x_new = rng.uniform(self.x_min, self.x_max, size=self.dim)
                    y_new = func(x_new)
                    calls_made += 1
                    self.X = np.vstack([self.X, x_new])
                    self.Y = np.append(self.Y, y_new)
                    if y_new < best_y:
                        best_y = y_new
                        best_x = x_new
                    continue
            
            # Adaptive Beta
            # Increase exploration slightly over time to avoid getting stuck
            # beta_t = 2.0 + 0.1 * log(calls_made)
            beta_t = 2.0 + 0.1 * np.log(calls_made + 1)
            
            # Generate candidates
            # Adaptive sampling box: shrink around best_x as we progress
            # Start with full box, shrink to 50% of the range around best_x
            progress = calls_made / self.budget
            # Shrink factor: 1.0 initially, down to 0.2
            shrink_factor = 1.0 - 0.8 * progress
            # But don't shrink too much, keep some exploration
            # Let's use a mixture: 50% random in full box, 50% in local box
            n_candidates = 1000
            n_local = n_candidates // 2
            n_global = n_candidates - n_local
            
            X_global = rng.uniform(self.x_min, self.x_max, size=(n_global, self.dim))
            
            # Local box around best_x
            # Radius based on shrink_factor
            # Original range is 10. Shrink to 10 * shrink_factor
            # But center at best_x
            radius_local = 5.0 * shrink_factor
            X_local = best_x + rng.uniform(-radius_local, radius_local, size=(n_local, self.dim))
            X_local = np.clip(X_local, self.x_min, self.x_max)
            
            X_cand = np.vstack([X_global, X_local])
            
            # Predict on candidates
            K_s = compute_kernel(X_cand, self.X) # (M, N)
            
            # Mean: K_s @ alpha * y_std + y_mean
            mu = (K_s @ alpha) * y_std + y_mean
            
            # Variance: sigma_f_sq - K_s @ K^-1 @ K_s.T
            # Use Cholesky for efficiency: K^-1 = L^-T L^-1
            # K_s @ L^-1 -> (M, N)
            v = np.linalg.solve(L, K_s.T) # (N, M)
            var = sigma_f_sq - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            # Acquisition: LCB = mu - beta * std
            # We want to minimize this.
            acq = mu - beta_t * std
            
            best_idx = np.argmin(acq)
            x_new = X_cand[best_idx]
            
            # Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # Update
            self.X = np.vstack([self.X, x_new])
            self.Y = np.append(self.Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new
                
        return best_x, best_y

# EVOLVE-BLOCK-END
