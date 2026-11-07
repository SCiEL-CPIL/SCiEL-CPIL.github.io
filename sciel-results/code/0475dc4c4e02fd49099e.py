import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            x = clip(x)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES global search
        # Use a population that adapts to the landscape
        # Start with a reasonable initial sigma
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # CMA-ES implementation (simplified, using scipy's Nelder-Mead as fallback isn't ideal)
        # Let's implement a basic CMA-ES or use a robust population-based approach
        
        # Actually, let's use a well-known approach: CMA-ES via scipy or manual
        # Since we can't rely on external CMA-ES packages, let's implement a simple one
        # Or better: use a combination of random restarts with local search
        
        # Strategy: 
        # 1. Do 30 random evaluations to find a good starting point
        # 2. Then do CMA-ES-like population search
        # 3. Then polish with Nelder-Mead or L-BFGS-B
        
        # Phase 1: Random initialization (20 evals)
        rng = np.random.default_rng(42)
        n_init = 20
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: CMA-ES population search
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim))
        lam = max(lam, 6)
        mu = lam // 2
        
        # Initialize population around best point found so far
        pop_x = [best_x + rng.normal(0, sigma0, dim) for _ in range(lam)]
        pop_f = [evaluate(x) for x in pop_x]
        
        # CMA-ES parameters
        sigma = sigma0
        step_size = 1.0
        mean = best_x.copy()
        
        # Simple adaptation: use the best half of the population
        while calls < budget and time.time() < deadline:
            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            # Update mean using weighted average of top mu
            weights = np.ones(mu) / mu
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * pop_x[sorted_idx[i]]
            
            # Update sigma based on spread of top points
            top_points = np.array([pop_x[sorted_idx[i]] for i in range(mu)])
            spread = np.mean(np.linalg.norm(top_points - new_mean, axis=1))
            sigma = max(0.01, spread * 0.5)
            
            # Generate new population
            pop_x = []
            pop_f = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                x = new_mean + rng.normal(0, sigma, dim)
                fx = evaluate(x)
                pop_x.append(x)
                pop_f.append(fx)
            
            if len(pop_x) < lam:
                break
            
            # Update best
            if min(pop_f) < best_f:
                pass  # Already updated in evaluate
            
            # Check for convergence
            if sigma < 0.01:
                break
            
            # Update mean
            mean = new_mean
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 3: Local polish with Nelder-Mead
        # Use remaining budget for local refinement
        remaining = budget - calls
        if remaining > 5:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for local optimization
                res = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining // 2, 'maxfev': remaining}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
