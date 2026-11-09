"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B Optimizer for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        # Define a wrapper for the function that tracks calls and updates best
        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Generate diverse starting points
        # We want to cover the space [-5, 5]^5 reasonably well.
        # Rosenbrock has a global minimum at (1, 1, ..., 1) in standard form.
        # With shifts/rotations, the minimum is somewhere in [-4, 4]^5.
        # We'll use a mix of deterministic and random starts.
        
        rng = np.random.default_rng(42)
        
        starts = []
        
        # 1. Theoretical minimum location (if shift is small)
        starts.append(np.ones(dim))
        
        # 2. Origin
        starts.append(np.zeros(dim))
        
        # 3. Grid-like points (corners and midpoints)
        # Too many grid points would exhaust budget. 
        # Let's pick a few strategic ones.
        # Corners
        for c1 in [-4, 4]:
            for c2 in [-4, 4]:
                p = np.full(dim, 0.0)
                p[0] = c1
                p[1] = c2
                starts.append(p)
                
        # 4. Random points
        # We have 500 calls. L-BFGS-B is efficient. 
        # Let's allow up to 10 restarts. 
        # Each restart might take 20-50 calls to converge.
        # 10 * 50 = 500. This is tight.
        # Let's be conservative with restarts.
        
        num_restarts = 8
        for _ in range(num_restarts):
            p = rng.uniform(-4, 4, dim)
            starts.append(p)

        # Deduplicate starts (approx)
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 1e-3:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Sort starts by heuristic? 
        # Points closer to (1,1,1,1,1) are more likely to be in the basin 
        # if the shift is not huge. But we don't know the shift.
        # Let's just evaluate them in order.
        
        for x0 in unique_starts:
            if time.time() > deadline or calls_used >= budget - 10:
                break
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    call_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 50, 'gtol': 1e-5, 'ftol': 1e-8}
                )
            except Exception:
                continue

            # If the result is better, update best (already done in call_func)
            # Check if we should continue
            if calls_used >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
