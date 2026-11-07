"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Bayesian-Optimization-like sampler with heavy local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: Random exploration to find a good starting point
        # Büche-Rastrigin is highly multimodal. We need to sample broadly.
        n_explore = min(100, budget // 5)
        
        best_x = None
        best_f = np.inf
        X = []
        F = []
        
        for i in range(n_explore):
            # Mix of uniform and grid-like sampling? 
            # Let's stick to uniform but maybe bias towards center? 
            # The optimum is in [-4, 4]^5. Uniform in [-5,5] is fine.
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            X.append(x)
            F.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local exploitation around the best point found
        # Use a Nelder-Mead or similar local optimizer, but we must count function calls.
        # Since we can't easily hook into scipy's internal call count without wrapping,
        # we will implement a simple gradient-free local search (like CMA-ES or Nelder-Mead manually)
        # or use scipy with a wrapper that stops at budget.
        
        # Actually, a simple approach: take the best point and run a local optimizer with a call limit.
        # Let's implement a small (1+1)-ES or Nelder-Mead locally.
        # Given the rugged landscape of Rastrigin, small steps are needed.
        
        # Let's use a simple local search: 
        # Start from best_x.
        # Try moving in random directions with decreasing step size.
        
        # Alternative: Use scipy.optimize.minimize with a method that allows step control?
        # Nelder-Mead is good for this.
        
        # We have budget - n_explore calls left.
        remaining = budget - n_explore
        
        # We will run a local Nelder-Mead optimization starting from best_x.
        # We need to count function calls carefully.
        
        class CallCounter:
            def __init__(self, func, limit):
                self.func = func
                self.limit = limit
                self.count = 0
                self.best_x = None
                self.best_f = np.inf
                self.history_x = []
                self.history_f = []
                
            def __call__(self, x):
                if self.count >= self.limit:
                    raise Exception("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                fx = self.func(x_clipped)
                self.count += 1
                self.history_x.append(x_clipped.copy())
                self.history_f.append(fx)
                if fx < self.best_f:
                    self.best_f = fx
                    self.best_x = x_clipped.copy()
                return fx

        # We want to use the remaining budget for local search.
        # Let's allocate most of it to local search, keeping a tiny bit for potential re-exploration?
        # No, once we have a good basin, local search is best.
        
        # However, Rastrigin has many local minima. If the first 100 points didn't hit the global basin, 
        # local search will get stuck. 
        # Maybe we should do a multi-start local search?
        
        # Strategy:
        # 1. Sample 50 points.
        # 2. Pick the best 3 points.
        # 3. Run local search from each of them, splitting the remaining budget.
        
        # Let's refine Phase 1:
        # Sample 50 points.
        # Sort by fitness.
        # Take top 3.
        # Run local search from each, using 1/3 of remaining budget for each.
        
        # Reset for this multi-start strategy
        n_initial = 50
        X = []
        F = []
        for i in range(n_initial):
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            X.append(x)
            F.append(fx)
            
        # Sort indices by fitness
        indices = np.argsort(F)
        top_k = min(3, len(indices))
        top_indices = indices[:top_k]
        
        remaining_budget = budget - n_initial
        
        # Split remaining budget among top_k starts
        budget_per_start = remaining_budget // top_k
        
        global_best_x = X[top_indices[0]]
        global_best_f = F[top_indices[0]]
        
        for k in range(top_k):
            start_x = X[top_indices[k]].copy()
            start_f = F[top_indices[k]]
            
            # Run local search
            # We'll use a simple (1+1)-ES for local search as it's robust and easy to budget
            sigma = 0.5
            x = start_x
            fx = start_f
            calls_used = 0
            max_calls = budget_per_start
            
            while calls_used < max_calls:
                # Generate a step
                # Adaptive sigma: if we improve, increase; else decrease
                step = sigma * rng.standard_normal(d)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        # Try a larger jump? Or break?
                        # If sigma is too small, we might be stuck.
                        # Let's try a random restart within a small neighborhood?
                        # Or just break and move to next start.
                        break
                        
        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
