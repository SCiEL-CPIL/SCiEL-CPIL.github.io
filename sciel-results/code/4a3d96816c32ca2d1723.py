"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B with multi-start strategy for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Generate diverse starting points
        starts = [np.zeros(dim)]
        # Add a point with small noise
        starts.append(rng.normal(0, 0.5, dim))
        # Add random points
        for _ in range(3):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Sort starts by distance to origin (heuristic)
        starts.sort(key=lambda x: np.linalg.norm(x))
        
        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            
            remaining = budget - call_count
            if remaining < 20:
                break
            
            # Limit function calls per start to ensure diversity
            # L-BFGS-B is efficient, 100-150 calls should be enough for local convergence
            max_calls_per_start = min(150, remaining)
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': max_calls_per_start, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                continue

        # Final polish with remaining budget
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 10:
                try:
                    res_final = minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': remaining, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 100}
                    )
                    if res_final.fun < best_f:
                        best_f = res_final.fun
                        best_x = res_final.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
