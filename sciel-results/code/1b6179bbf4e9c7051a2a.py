import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start Nelder-Mead with adaptive simplex scaling for Bent Cigar.
    
    Strategy:
    - Phase 1: Multi-start Nelder-Mead from diverse points (including origin)
    - Phase 2: Final polish with the best solution found
    - Uses tracked function calls to stay within budget
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = [0]  # Mutable counter for closures

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            if calls[0] >= budget:
                raise Exception("budget exceeded")
            calls[0] += 1
            return func(np.clip(x, -5, 5))

        rng = np.random.default_rng(42)
        
        best_x = None
        best_fx = np.inf

        # Generate diverse starting points
        # Include origin and points spread across the domain
        n_starts = 6
        starts = []
        
        # Origin is a good starting point for symmetric problems
        starts.append(np.zeros(dim))
        
        # Add diverse random points
        for i in range(1, n_starts):
            pt = rng.uniform(-4, 4, dim)
            starts.append(pt)
        
        # Allocate budget: give each start a portion, with more to earlier starts
        # Reserve some budget for final polish
        polish_budget = 30
        available = budget - polish_budget
        budget_per_start = max(15, available // n_starts)
        
        for start_idx, start_point in enumerate(starts):
            if time.time() >= deadline:
                break
            if calls[0] >= budget:
                break
            
            remaining_budget = budget - calls[0]
            if remaining_budget < 10:
                break
            
            # Use Nelder-Mead from this starting point
            try:
                result = minimize(
                    tracked_func,
                    start_point,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_budget, 200),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                # If Nelder-Mead fails, try a simple local search
                try:
                    local_result = minimize(
                        tracked_func,
                        start_point,
                        method='Nelder-Mead',
                        options={'maxiter': min(remaining_budget, 50), 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if local_result.fun < best_fx:
                        best_fx = local_result.fun
                        best_x = local_result.x.copy()
                except Exception:
                    pass
        
        # Final polish with the best solution found
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 5:
                try:
                    final_result = minimize(
                        tracked_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 100),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if final_result.fun < best_fx:
                        best_fx = final_result.fun
                        best_x = final_result.x.copy()
                except Exception:
                    pass
        
        # Ensure we have a valid result
        if best_x is None:
            # Fallback: use the first function evaluation
            x0 = np.zeros(dim)
            best_x = x0
            best_fx = func(x0)
            calls[0] = 1
        
        return best_x, best_fx
