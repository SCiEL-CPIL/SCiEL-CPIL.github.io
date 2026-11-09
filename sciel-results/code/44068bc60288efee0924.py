"""Hybrid multi-restart optimizer: ES exploration + L-BFGS-B refinement for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer combining (1+1)-ES exploration with L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: (1+1)-ES exploration to find a good basin
        # Use ~60% of budget for exploration
        es_budget = int(budget * 0.6)
        rng = np.random.default_rng(42)
        
        # Start from a diverse point
        x = rng.uniform(-4, 4, dim)
        fx = safe_func(x)
        update_best(x, fx)
        
        sigma = 2.0
        for i in range(es_budget - 1):
            if time.time() > deadline or call_count >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = safe_func(y)
            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.3
            else:
                sigma *= 0.85
        
        # Phase 2: Multi-restart L-BFGS-B from the ES result and diverse points
        # Use remaining budget for local refinement
        remaining_budget = budget - call_count
        if remaining_budget < 10:
            remaining_budget = max(10, budget - call_count)
        
        # Generate starting points for L-BFGS-B
        starts = []
        if best_x is not None:
            starts.append(best_x.copy())
        
        # Add the ES trajectory endpoints and some random points
        starts.append(x.copy())
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Shuffle
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]
        
        # Allocate remaining budget among restarts
        maxfev_per_restart = max(10, remaining_budget // len(starts))
        
        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfev': maxfev_per_restart}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass
        
        # Phase 3: Final refinement from the best point with remaining budget
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 5:
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxfev': remaining}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
