"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global-local optimizer: random restarts + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # We'll track the best solution found
        best_x = None
        best_f = np.inf

        # Phase 1: Global exploration with multiple random starts
        # Use about 40% of budget for global search, 60% for local refinement
        num_global = 40
        num_local = budget - num_global

        # Generate initial candidate points
        # Mix of uniform random and some structured points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(n))
        
        # Random points in the box
        for _ in range(num_global - 1):
            candidates.append(rng.uniform(-5, 5, n))
        
        # Evaluate all candidates
        for x in candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using Nelder-Mead from the best points
        # Sort candidates by function value and refine the top ones
        eval_results = []
        for x in candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            eval_results.append((fx, x))
        
        eval_results.sort(key=lambda t: t[0])

        # Refine top candidates with Nelder-Mead
        num_refine = min(10, len(eval_results))
        calls_used = num_global  # approximate
        calls_remaining = budget - calls_used

        for i in range(num_refine):
            if time.time() > deadline:
                break
            if calls_remaining <= 0:
                break

            fx_start, x_start = eval_results[i]
            
            # Limit function calls for Nelder-Mead
            max_iter = max(20, calls_remaining // (num_refine - i))
            max_iter = min(max_iter, 200)
            
            try:
                result = minimize(
                    func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                calls_used += result.nfev
                calls_remaining = budget - calls_used
            except Exception:
                pass

        # Phase 3: If we still have budget, do more random restarts + quick refinement
        while calls_remaining > 10 and time.time() < deadline:
            x_try = rng.uniform(-5, 5, n)
            fx_try = func(x_try)
            calls_used += 1
            calls_remaining -= 1
            
            if fx_try < best_f:
                best_f = fx_try
                best_x = x_try.copy()
                
                # Quick local refinement
                if calls_remaining > 20 and time.time() < deadline:
                    try:
                        result = minimize(
                            func,
                            x_try,
                            method='Nelder-Mead',
                            options={'maxiter': min(calls_remaining - 5, 100), 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                        calls_used += result.nfev
                        calls_remaining = budget - calls_used
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
