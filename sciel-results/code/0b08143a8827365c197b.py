"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with deterministic grid initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Store top candidates for local refinement
        top_candidates = [] # List of (f, x)
        max_candidates = 3

        # Helper to count calls and check deadline
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget exceeded")
            
            calls_made += 1
            f_val = func(x)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy() if isinstance(x, np.ndarray) else x
            
            # Maintain top candidates
            if len(top_candidates) < max_candidates:
                top_candidates.append((f_val, x.copy() if isinstance(x, np.ndarray) else x))
                top_candidates.sort(key=lambda item: item[0])
            else:
                if f_val < top_candidates[-1][0]:
                    top_candidates.pop()
                    top_candidates.append((f_val, x.copy() if isinstance(x, np.ndarray) else x))
                    top_candidates.sort(key=lambda item: item[0])
                    
            return f_val

        # Phase 1: Deterministic Grid Search
        # Use a 3^5 grid if budget allows
        grid_dim = 3
        if grid_dim ** self.dim <= self.budget:
            grid_axes = [np.linspace(-5, 5, grid_dim) for _ in range(self.dim)]
            grid_points = np.array(np.meshgrid(*grid_axes)).reshape(self.dim, -1).T
            
            for p in grid_points:
                if calls_made >= self.budget:
                    break
                try:
                    counted_func(p)
                except RuntimeError:
                    break

        # Phase 2: Local Refinement using Nelder-Mead
        # Use remaining budget for local optimization on top candidates
        remaining_budget = self.budget - calls_made
        if remaining_budget > 10 and top_candidates:
            # Split budget among candidates
            n_starts = min(len(top_candidates), 3)
            budget_per_start = remaining_budget // n_starts
            
            for i in range(n_starts):
                if time.time() > deadline - 10:
                    break
                if calls_made >= self.budget:
                    break
                
                f_init, x_init = top_candidates[i]
                
                # Estimate maxiter for Nelder-Mead
                # Calls per iteration ~ dim + 1
                maxiter = max(1, (budget_per_start - 1) // (self.dim + 1))
                
                try:
                    res = minimize(counted_func, x_init, method='Nelder-Mead',
                                   options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
                    # best_f and best_x are updated in counted_func
                except RuntimeError:
                    break

        # Fallback if no calls were made
        if calls_made == 0:
            x0 = np.random.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
