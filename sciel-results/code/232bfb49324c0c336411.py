"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_f = float('inf')
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # Phase 1: Multi-start L-BFGS-B
        # We use a fraction of the budget for multi-start, and the rest for refinement
        # Since L-BFGS-B is fast, we can do many starts.
        # Let's try up to 50 starts, using remaining budget if time permits.
        
        # First, generate a set of starting points
        # 1. Zero
        # 2. Random points
        # 3. Grid points (coarse)
        
        starts = []
        starts.append(np.zeros(dim))
        
        # Add random points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Add some structured points based on common BBOB shifts
        # Often shifts are around 0 or specific values.
        # Let's add points near boundaries and center.
        for i in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                p = np.zeros(dim)
                p[i] = val
                starts.append(p)
        
        # Deduplicate starts
        unique_starts = []
        seen = set()
        for s in starts:
            key = tuple(s.round(5))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        
        # Sort starts: zero first, then others
        unique_starts.sort(key=lambda x: np.sum(x**2))
        
        for x0 in unique_starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
                
            # Define wrapper to count calls and check time
            def wrapped_func(x):
                nonlocal call_count
                if call_count >= budget or time.time() > deadline:
                    return float('inf')
                call_count += 1
                return func(x)
            
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # If we've used a lot of budget, stop
            if call_count > budget * 0.8:
                break
        
        # Phase 2: Refinement with (1+1)-ES or Nelder-Mead on the best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            # Use Nelder-Mead for final polish, it's robust for non-smooth or noisy
            # But Rosenbrock is smooth. L-BFGS-B should be good.
            # Let's do a few more L-BFGS-B runs from the best point with tighter tol
            # Or just do a simple local search.
            
            # Let's try a few more L-BFGS-B runs from best_x with different initial steps?
            # Or just rely on the previous ones.
            
            # Let's do a short (1+1)-ES to polish
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(100, budget - call_count)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                if call_count >= budget:
                    break
                call_count += 1
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x, fx
                    sigma *= 1.2
                else:
                    sigma *= 0.8
        else:
            # If no best_x found (shouldn't happen), do a random search
            x = rng.uniform(-5, 5, dim)
            if call_count < budget:
                call_count += 1
                best_f = func(x)
                best_x = x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
