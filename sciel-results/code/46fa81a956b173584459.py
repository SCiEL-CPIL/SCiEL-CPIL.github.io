import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Global Search using Latin Hypercube Sampling (LHS)
        # BBOB-21 has many local optima. We need a dense global search to locate
        # the basin of the global optimum. We allocate 40% of the budget here.
        n_global = int(budget * 0.4)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        # Sort by fitness
        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # Phase 2: Local Refinement using Nelder-Mead with aggressive restarts
        # Nelder-Mead is a derivative-free simplex method that works well on
        # smooth multimodal landscapes when given good initial points.
        
        n_local_candidates = 8
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        budget_per_local = max(15, remaining_budget // n_local_candidates)
        
        # Track call count for Nelder-Mead
        call_counter = [0]
        
        def wrapped_func(x):
            if call_counter[0] >= budget - calls or time.time() > deadline:
                return float('inf')
            val = func(x)
            call_counter[0] += 1
            return val
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best global sample
            x0 = lhs[sorted_indices[k]].copy()
            f0 = f_values[sorted_indices[k]]
            
            # Nelder-Mead with multiple restarts at different scales
            # Try different initial simplex sizes to handle varying basin widths
            simplex_sizes = [0.3, 0.6, 1.0, 2.0]
            
            for simplex_size in simplex_sizes:
                if calls >= budget or time.time() > deadline:
                    break
                
                # Set up Nelder-Mead
                # maxiter controls number of iterations, each iteration does ~dim+1 evals
                # We want to use about budget_per_local / len(simplex_sizes) calls per restart
                max_iter = max(5, budget_per_local // (len(simplex_sizes) * (dim + 1)))
                
                try:
                    result = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'xatol': 1e-4,
                            'fatol': 1e-6,
                            'maxiter': max_iter,
                            'adaptive': True
                        }
                    )
                    
                    # Update best if improved
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        # Use the new best as starting point for next restart
                        x0 = result.x.copy()
                        f0 = result.fun
                    elif result.fun < f0:
                        # At least improved from this starting point
                        f0 = result.fun
                        x0 = result.x.copy()
                        
                except Exception:
                    pass
                
                # Update calls
                calls += call_counter[0]
                call_counter[0] = 0
                
                # If no improvement, try next simplex size
                if time.time() > deadline:
                    break

        return best_x, best_f
