import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell's method for Bent Cigar optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = [0]  # Use list for mutable counter in closures

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            if calls[0] >= budget:
                raise Exception("budget exceeded")
            calls[0] += 1
            return func(np.clip(x, -5, 5))

        # Multi-start Powell's method
        # Strategy: Use multiple starting points spread across the domain
        # Powell's method is derivative-free and handles anisotropic landscapes well
        
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        n_starts = 8
        starts = []
        
        # Center point
        starts.append(np.zeros(dim))
        
        # Grid-like points
        for i in range(1, n_starts):
            # Spread points across [-4, 4] to avoid boundary issues
            pt = rng.uniform(-4, 4, dim)
            starts.append(pt)
        
        best_x = None
        best_fx = np.inf
        
        # Allocate budget proportionally to each start
        # Give more budget to earlier starts as they may find good solutions
        budget_per_start = max(10, budget // n_starts)
        
        for start_idx, start_point in enumerate(starts):
            if time.time() >= deadline:
                break
            if calls[0] >= budget:
                break
            
            # Track calls for this specific Powell run
            remaining_budget = budget - calls[0]
            if remaining_budget < 10:
                break
            
            def powell_func(x, _calls=calls):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if _calls[0] >= budget:
                    raise Exception("budget exceeded")
                _calls[0] += 1
                return func(np.clip(x, -5, 5))
            
            try:
                result = minimize(
                    powell_func,
                    start_point,
                    method='Powell',
                    options={
                        'maxiter': min(remaining_budget, 500),
                        'maxfev': min(remaining_budget, 500),
                        'xtol': 1e-10,
                        'ftol': 1e-12
                    }
                )
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                # If Powell fails, try a simple local search from the start point
                try:
                    local_result = minimize(
                        powell_func,
                        start_point,
                        method='Nelder-Mead',
                        options={'maxiter': min(remaining_budget, 100), 'xatol': 1e-10, 'fatol': 1e-12}
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
                def final_func(x, _calls=calls):
                    if time.time() >= deadline:
                        raise Exception("timeout")
                    if _calls[0] >= budget:
                        raise Exception("budget exceeded")
                    _calls[0] += 1
                    return func(np.clip(x, -5, 5))
                
                try:
                    final_result = minimize(
                        final_func,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': min(remaining, 100),
                            'maxfev': min(remaining, 100),
                            'xtol': 1e-12,
                            'ftol': 1e-14
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
