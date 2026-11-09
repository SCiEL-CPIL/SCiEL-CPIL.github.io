import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f, deadline
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            
            # Ensure x is a valid vector
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                raise ValueError("Wrong dimension")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite input")
                
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Random Scan
        # 50 random points is a good balance between exploration and cost
        n_scan = 50
        for _ in range(n_scan):
            try:
                x = rng.uniform(-5, 5, self.dim)
                fx = wrapped_func(x)
            except (TimeoutError, ValueError):
                break

        if best_x is None:
            # Fallback if even the first call failed (should not happen)
            return np.zeros(self.dim), np.inf

        # Phase 2: Local Optimization with Nelder-Mead
        # We use the best point from the scan as the initial guess
        # Nelder-Mead is derivative-free and robust for low dimensions
        
        try:
            res = minimize(
                wrapped_func, 
                best_x, 
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'maxiter': 1000,
                    'maxfev': self.budget - calls_made # Hard cap on function evaluations
                }
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except (TimeoutError, ValueError):
            pass
        except Exception:
            # Catch any other exceptions to ensure we return something
            pass

        # Phase 3: If time and budget remain, try a few perturbations
        # This can help if the first Nelder-Mead run got stuck in a local minimum
        # (though f14 is likely unimodal in the transformed space)
        while time.time() < deadline - 5.0 and calls_made < self.budget - 10:
            # Perturb the best solution
            sigma = 0.1
            x_pert = best_x + rng.normal(0, sigma, self.dim)
            x_pert = np.clip(x_pert, -5, 5)
            try:
                res2 = minimize(
                    wrapped_func,
                    x_pert,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 50,
                        'maxfev': min(50, self.budget - calls_made)
                    }
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                break

        return best_x, best_f
