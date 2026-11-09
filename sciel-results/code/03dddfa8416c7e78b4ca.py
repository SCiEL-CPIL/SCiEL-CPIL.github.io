"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Pre-compute some strategic starting points
        # For Rosenbrock, the minimum is at (1,1,...,1) in standard form,
        # but here it's rotated and shifted. We try several diverse starts.
        
        rng = np.random.default_rng(42)
        
        # Strategy: Use a population-based approach with adaptive sigma
        # We'll implement a simple CMA-ES-like or (mu+lambda)-ES
        
        # Let's try a more robust approach:
        # 1. Initial random restarts to find a good basin
        # 2. Then refine with a local optimizer
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Random restarts with small local search
        # Use about 70% of budget for exploration, 30% for exploitation
        
        # Actually, let's use a smarter approach:
        # Implement a simplified CMA-ES or use scipy's differential_evolution
        # But we need to be careful about function call budget.
        
        # Let's implement a (1+1)-ES with better step size adaptation
        # and multiple restarts
        
        def run_es(x0, max_calls, rng):
            """Run (1+1)-ES from x0 with max_calls function evaluations."""
            x = x0.copy()
            fx = func(x)
            sigma = 1.0
            calls = 1
            success_count = 0
            window = 10
            successes_in_window = 0
            
            for _ in range(max_calls - 1):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    successes_in_window += 1
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                # Adapt sigma based on success rate in last window
                # Simple adaptation: if too many successes, increase sigma
                # if too few, decrease sigma
                if successes_in_window > window * 0.7:
                    sigma *= 1.1
                    successes_in_window = 0
                elif successes_in_window < window * 0.2:
                    sigma *= 0.9
                    successes_in_window = 0
                
                if calls >= max_calls:
                    break
            
            return x, fx, calls
        
        # Multiple restart strategy
        n_restarts = 5
        calls_per_restart = (budget // n_restarts) - 10  # reserve some for polish
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Generate diverse starting points
            if r == 0:
                x0 = rng.uniform(-3, 3, dim)
            elif r == 1:
                x0 = rng.uniform(-5, 5, dim)
            elif r == 2:
                # Try near the center
                x0 = rng.normal(0, 1.5, dim)
            elif r == 3:
                # Try a point biased toward positive values (Rosenbrock min at 1,1,...)
                x0 = rng.uniform(0, 3, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            x, fx, used_calls = run_es(x0, calls_per_restart, rng)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Polish with Nelder-Mead
        if best_x is not None and time.time() < deadline - 30:
            # Use Nelder-Mead to refine
            remaining_calls = budget - 1 - sum(1 for _ in range(n_restarts))  # rough estimate
            # Actually, we need to track calls more carefully
            
            # Let's just try Nelder-Mead with a small maxiter
            try:
                # Create a wrapper that counts calls
                call_count = [0]
                original_func = func
                
                def counting_func(x):
                    call_count[0] += 1
                    if call_count[0] > 50:  # Limit polish calls
                        return np.inf
                    return original_func(x)
                
                res = minimize(counting_func, best_x, method='Nelder-Mead',
                              options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
