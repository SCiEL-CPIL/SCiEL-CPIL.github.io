"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start CMA-ES style search with L-BFGS-B polish for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        calls_used = 0
        best_x = None
        best_f = np.inf

        def try_eval(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget or time.time() > deadline:
                return None
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diversified initial sampling
        n_initial = min(20, budget)
        # Mix of uniform, Gaussian, and structured points
        for i in range(n_initial):
            if time.time() > deadline or calls_used >= budget:
                break
            if i < 5:
                x = rng.uniform(-5, 5, n)
            elif i < 10:
                x = rng.normal(0, 2, n)
                x = np.clip(x, -5, 5)
            elif i < 15:
                # Sparse structured points
                x = rng.choice([-4, -2, 0, 2, 4], n)
            else:
                x = rng.uniform(-5, 5, n)
            f = try_eval(x)
            if f is None:
                break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Iterative local optimization from diverse starts
        # Use L-BFGS-B from multiple starting points
        n_starts = min(30, budget - calls_used - 50)
        
        # Generate diverse starting points
        starts = []
        # Best point and its neighborhood
        for _ in range(5):
            perturbation = rng.normal(0, 0.5, n)
            x = np.clip(best_x + perturbation, -5, 5)
            starts.append(x)
        
        # Random points
        for _ in range(min(10, n_starts - 5)):
            x = rng.uniform(-5, 5, n)
            starts.append(x)
        
        # Gaussian points
        for _ in range(min(10, n_starts - 15)):
            x = np.clip(rng.normal(0, 2, n), -5, 5)
            starts.append(x)
        
        # Structured points based on Rosenbrock valley
        for _ in range(min(5, n_starts - 20)):
            # Try to be near the valley: x_i ≈ x_{i-1}^2
            x = rng.uniform(-2, 2, n)
            for i in range(1, n):
                x[i] = x[i-1]**2
            x = np.clip(x, -5, 5)
            starts.append(x)

        # Run L-BFGS-B from each start
        for i, x0 in enumerate(starts):
            if time.time() > deadline or calls_used >= budget - 20:
                break
            
            try:
                res = minimize(
                    func,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-8}
                )
                calls_used += res.nfev
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
                    # If significant improvement, do additional polish
                    if calls_used < budget - 50 and time.time() < deadline:
                        try:
                            res2 = minimize(
                                func,
                                x0=best_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * n,
                                options={'maxiter': 500, 'maxfun': 20000, 'gtol': 1e-10}
                            )
                            calls_used += res2.nfev
                            if res2.fun < best_f:
                                best_f = res2.fun
                                best_x = res2.x.copy()
                        except Exception:
                            pass
            except Exception:
                pass

        # Final polish with remaining budget
        if best_x is not None and calls_used < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    func,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 1000, 'maxfun': 100000, 'gtol': 1e-12}
                )
                calls_used += res_final.nfev
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
