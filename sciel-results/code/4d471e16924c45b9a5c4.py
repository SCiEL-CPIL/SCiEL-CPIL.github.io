"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """SLSQP-based optimizer with random restarts for Rosenbrock-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Initial random exploration
        # We need to find a good starting point for SLSQP.
        # Rosenbrock is narrow, so random points are usually bad.
        # We will use a small number of random points to seed the local search.
        n_random = 5
        candidates = []
        
        for _ in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: SLSQP local optimization from promising starts
        # SLSQP can be more robust to the narrow valley if it takes good steps,
        # but it is sensitive to initialization. 
        # We will run SLSQP on the top few candidates.
        
        n_local = min(5, len(candidates))
        
        for i in range(n_local):
            if time.time() > deadline - 10 or calls_made >= budget - 5:
                break

            fx, x0 = candidates[i]
            
            # If the initial point is already very good, skip or do a short run?
            # SLSQP requires a reasonable start.
            
            try:
                # Use SLSQP with finite difference gradients
                # maxiter is limited by budget, but SLSQP uses function calls for gradients
                # Each iteration uses ~2*dim function calls for gradient estimation.
                # So maxiter should be roughly (budget - calls_made) / (2*dim)
                
                remaining_budget = budget - calls_made
                max_iter_est = max(10, remaining_budget // (2 * dim + 1))
                
                result = minimize(
                    safe_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_est,
                        'ftol': 1e-12,
                        'disp': False
                    }
                )
                
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
                    
            except Exception:
                pass

        # Phase 3: If budget remains, try more random restarts with SLSQP
        while calls_made < budget - 10 and time.time() < deadline - 10:
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Try to refine this new random point
            if calls_made < budget - 5:
                try:
                    remaining_budget = budget - calls_made
                    max_iter_est = max(5, remaining_budget // (2 * dim + 1))
                    
                    result = minimize(
                        safe_func,
                        x,
                        method='SLSQP',
                        bounds=bounds,
                        options={
                            'maxiter': max_iter_est,
                            'ftol': 1e-12,
                            'disp': False
                        }
                    )
                    
                    if np.isfinite(result.fun) and result.fun < best_f:
                        best_f = float(result.fun)
                        best_x = np.clip(result.x, -5, 5).copy()
                        
                except Exception:
                    pass

        # Phase 4: Final high-precision polish on the best found
        if best_x is not None and calls_made < budget - 2 and time.time() < deadline - 2:
            try:
                remaining_budget = budget - calls_made
                # Allow more iterations for polish if budget allows
                max_iter_est = max(20, remaining_budget // (2 * dim))
                
                result = minimize(
                    safe_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_est,
                        'ftol': 1e-15,
                        'disp': False
                    }
                )
                
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
                    
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
