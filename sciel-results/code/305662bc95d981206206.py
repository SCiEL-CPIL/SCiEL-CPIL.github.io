import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = np.clip(np.asarray(x, dtype=np.float64), -5, 5).copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)
        bounds = [(-5, 5)] * dim

        # Phase 1: Global Exploration with Differential Evolution
        # Use a small population and few iterations to conserve budget
        # popsize=5, maxiter=10 -> ~500 calls worst case, but we limit budget
        # We want to spend about 60% of budget here.
        de_budget = int(budget * 0.6)
        
        # DE requires a function that returns a scalar.
        # We must ensure DE does not exceed our budget.
        # DE's internal loop is hard to interrupt easily without a custom callback or wrapper.
        # The wrapper `safe_func` will stop returning new values (returning best_f) 
        # once budget is hit, which might cause DE to converge prematurely or error out?
        # Actually, if `safe_func` returns the same value, DE might think it converged.
        # It's safer to run DE with a strict maxiter and popsize that fits in de_budget.
        
        # Estimate calls for DE:
        # Initial population: popsize * dim
        # Each generation: popsize * dim (approx, depends on mutation)
        # Let's use popsize=4, maxiter=15.
        # Calls ~ 4*5 + 15*4*5 = 20 + 300 = 320.
        # This is safe for budget 500.
        
        try:
            # Use a custom function wrapper for DE that also tracks calls
            # But `safe_func` already does that.
            # However, DE might call the function with vectors outside bounds?
            # `differential_evolution` clips to bounds.
            
            # We need to be careful: DE might use more calls than expected if it doesn't converge.
            # Let's limit maxiter strictly.
            
            de_result = differential_evolution(
                safe_func,
                bounds,
                popsize=4,
                maxiter=15,
                tol=1e-6,
                atol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=False,  # We will do our own polish
                updating='immediate'
            )
            
            if np.isfinite(de_result.fun) and de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x
        except Exception:
            pass

        # Phase 2: Local Optimization from best point found so far
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline - 20:
            # Use L-BFGS-B for fast local convergence
            try:
                local_result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': budget - calls_made - 5,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )
                if np.isfinite(local_result.fun) and local_result.fun < best_f:
                    best_f = local_result.fun
                    best_x = local_result.x
            except Exception:
                pass

        # Phase 3: If we have budget left, try a few random restarts + L-BFGS-B
        # This helps if DE got stuck in a local minimum (unlikely for Rosenbrock but possible for other BBOB)
        # Or if DE didn't explore enough.
        if calls_made < budget - 50 and time.time() < deadline - 30:
            n_restarts = 5
            for _ in range(n_restarts):
                if calls_made >= budget - 20 or time.time() > deadline - 10:
                    break
                x0 = rng.uniform(-5, 5, dim)
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'maxfun': max(10, (budget - calls_made) // 10),
                            'ftol': 1e-12,
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Final Polish
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                final_result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(5, budget - calls_made - 1),
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                    }
                )
                if np.isfinite(final_result.fun) and final_result.fun < best_f:
                    best_f = final_result.fun
                    best_x = final_result.x
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = safe_func(best_x)
            else:
                best_f = 0.0

        return best_x, best_f
