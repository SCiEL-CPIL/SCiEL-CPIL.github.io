"""Surrogate-assisted optimization with Gaussian Process and EI for Rastrigin rotated f15."""
import numpy as np
import time
import math
from scipy.stats import norm

class Optimizer:
    """Surrogate-assisted optimizer for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Data storage
        X = []
        Y = []
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        # Phase 1: Initial sampling
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_used += 1
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        X = np.array(X)
        Y = np.array(Y)

        # Phase 2: Surrogate-assisted search
        # GP parameters
        sigma_f = 1.0 # Signal variance
        length_scale = 1.0 # Length scale
        noise = 1e-6
        
        # We will adapt these or just keep them fixed for simplicity? 
        # Fixed is safer for speed. 
        # Let's use a standard RBF kernel.
        
        def kernel(x1, x2):
            # x1: (N, dim), x2: (M, dim)
            d2 = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2 * np.dot(x1, x2.T)
            d2 = np.maximum(d2, 0)
            return sigma_f**2 * np.exp(-d2 / (2 * length_scale**2))

        def predict_gp(x_new):
            # x_new: (M, dim)
            K_xx = kernel(X, X)
            K_xx += noise * np.eye(len(X))
            K_xn = kernel(X, x_new)
            
            # Solve for weights
            try:
                L = np.linalg.cholesky(K_xx)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            except:
                # Fallback to identity if singular
                alpha = Y.copy()
                L = np.eye(len(X))
            
            # Predictions
            mu = K_xn.T @ alpha
            var = sigma_f**2 - np.sum(K_xn * np.linalg.solve(L.T, K_xn.T), axis=0)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            return mu, sigma

        # Acquisition Function: Expected Improvement
        def expected_improvement(x_new, current_best):
            mu, sigma = predict_gp(x_new)
            z = (current_best - mu) / sigma
            ei = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Optimization of EI
        # We will use a multi-start local search on the EI function.
        # Since EI is not differentiable, we use Nelder-Mead or simple random perturbations.
        # Given dim=5, a simple local search or a few random starts + local refinement is good.
        
        n_ei_candidates = 10 # Number of random starts for EI optimization
        local_steps = 20     # Steps for local search on EI
        
        remaining_budget = budget - calls_used
        
        # We need to be careful with the number of calls used for EI optimization.
        # The EI function calls predict_gp, which is O(N^3) or O(N^2). 
        # With N=500, this is ~125 million ops per prediction if we do it naively.
        # 500 steps * 10 candidates * 20 local steps * (N^2) ... might be too slow.
        
        # Optimization: 
        # 1. Reduce N by keeping only the best 100 points? No, GP needs global info.
        # 2. Use a faster kernel or approximation?
        # 3. Reduce frequency of GP refitting? No, we update every step.
        
        # Let's estimate speed.
        # N=500. Cholesky is O(N^3/3) = 500^3/3 ~ 4e7 flops.
        # 500 iterations * 4e7 = 2e10 flops.
        # Modern CPU does ~1e10 flops/sec. So ~2 seconds for Cholesky alone? 
        # Plus matrix multiplications.
        # It might be borderline for 300s.
        
        # To be safe, let's limit the GP size to the best 100 points or use a simpler model.
        # Or, just use a simpler surrogate: K-Nearest Neighbors with a distance-weighted mean?
        # The directive suggests GP or Random Forest. 
        # Random Forest is faster to update (O(N log N) or O(N) for prediction if pre-sorted).
        # But GP is more standard for continuous optimization.
        
        # Let's try to optimize the GP code.
        # Instead of refitting the whole GP every step, we can just add the new point.
        # But the Cholesky factorization needs to be updated or redone.
        # Redoing it is O(N^3).
        
        # Alternative: Use a simpler acquisition function that doesn't require full GP?
        # Or use a **Random Forest** surrogate.
        # Random Forest prediction is O(T * N * log N) or similar.
        # Building a new tree every step is expensive.
        
        # Let's stick to GP but optimize the loop.
        # If N grows large, we can subsample the training data for the GP, keeping the best points and some random ones to maintain diversity.
        
        max_gp_size = 100
        
        for step in range(remaining_budget):
            if time.time() > deadline or calls_used >= budget:
                break
            
            # Subsample if X is too large
            if len(X) > max_gp_size:
                # Keep best 50 and random 50
                indices = np.argsort(Y)[:50]
                random_indices = rng.choice(len(X), size=50, replace=False)
                keep_indices = np.unique(np.concatenate([indices, random_indices]))
                X = X[keep_indices]
                Y = Y[keep_indices]
            
            # Optimize EI
            # Generate random starts
            starts = rng.uniform(-4.5, 4.5, (n_ei_candidates, dim))
            
            best_ei_val = -np.inf
            best_x_ei = None
            
            for s in starts:
                # Local search around s
                x_cur = s.copy()
                ei_cur = expected_improvement(x_cur.reshape(1, -1), best_f)[0]
                
                # Simple local search: random perturbations
                for ls in range(local_steps):
                    x_next = x_cur + rng.normal(0, 0.5, dim)
                    x_next = np.clip(x_next, -4.5, 4.5)
                    ei_next = expected_improvement(x_next.reshape(1, -1), best_f)[0]
                    if ei_next > ei_cur:
                        x_cur = x_next
                        ei_cur = ei_next
                
                if ei_cur > best_ei_val:
                    best_ei_val = ei_cur
                    best_x_ei = x_cur
            
            if best_x_ei is None:
                break
                
            # Evaluate the best EI point
            fx = func(best_x_ei)
            calls_used += 1
            X = np.vstack([X, best_x_ei])
            Y = np.append(Y, fx)
            
            if fx < best_f:
                best_f = fx
                best_x = best_x_ei.copy()

        # Phase 3: Local polish
        if time.time() < deadline and calls_used < budget:
            # Use Nelder-Mead or simple local random search
            # Let's do a simple local random search with decreasing step size
            step_size = 0.5
            for i in range(min(50, budget - calls_used)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, step_size / dim**0.5, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    step_size *= 0.9
                else:
                    step_size *= 0.95

        return (best_x, best_f)
