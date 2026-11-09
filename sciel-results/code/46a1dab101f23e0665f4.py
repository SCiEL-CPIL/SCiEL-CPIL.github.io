from scipy.optimize import minimize
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Halton-like sampling
        # We use a Halton sequence for better distribution in low dimensions
        def halton_sequence(n_points, dim):
            points = np.zeros((n_points, dim))
            for i in range(1, n_points + 1):
                for j in range(dim):
                    base = j + 2 # Primes: 2, 3, 5, ...
                    f = 1.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        i_val = i_val // base
                    points[i-1, j] = f
            return points

        # Use 100 initial points to get good coverage
        n_init = 100
        if budget < n_init + 20:
             n_init = max(10, budget // 2)
             
        # Generate Halton points in [0, 1)^d and map to [-5, 5]^d
        # Halton sequence is in [0, 1). 
        # To avoid boundary issues, we can scale slightly or just map.
        # Standard mapping: x = -5 + 10 * h
        halton_pts = halton_sequence(n_init, n)
        
        # Add a few random points to ensure robustness against any specific Halton weaknesses
        rng = np.random.default_rng(42)
        n_random = 10
        random_pts = rng.uniform(-5.0, 5.0, (n_random, n))
        
        all_init_pts = np.vstack([halton_pts, random_pts])
        
        # Evaluate all initial points
        init_results = []
        for i in range(len(all_init_pts)):
            if calls >= budget or time.time() > deadline:
                break
            x = all_init_pts[i]
            f = try_func(x)
            init_results.append((f, x))
            
        if not init_results:
            return best_x, best_f

        # Phase 2: Multi-start L-BFGS-B
        # Sort by function value
        init_results.sort(key=lambda x: x[0])
        
        # Select top K candidates for local optimization
        # K depends on remaining budget. Each L-BFGS-B run might take ~50-100 calls.
        # Let's aim for K=5 to K=10 restarts.
        max_restarts = 10
        K = min(max_restarts, len(init_results))
        
        # Allocate remaining budget among restarts
        remaining_calls = budget - calls
        if remaining_calls <= 0:
            return best_x, best_f
            
        # Roughly equal split, but ensure at least some minimum
        # L-BFGS-B is fast, so 50 calls per restart is reasonable for 5D
        calls_per_restart = max(10, remaining_calls // K)
        
        # We will run L-BFGS-B from the top K points
        # We need to be careful with the deadline
        for k in range(K):
            if calls >= budget or time.time() > deadline:
                break
                
            start_x = init_results[k][1].copy()
            
            def obj(x):
                return try_func(x)
            
            try:
                # Limit maxiter to avoid hanging if it doesn't converge
                # maxfun is limited by our try_func budget check, but maxiter helps stop early
                # We don't know exactly how many function evals L-BFGS-B will do, 
                # but try_func enforces the global budget.
                res = minimize(obj, start_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
