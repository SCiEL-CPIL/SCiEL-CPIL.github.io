"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start differential evolution with gradient-based polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        call_count = 0

        def timed_func(x):
            nonlocal call_count
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            return float(func(x))

        # Phase 1: Differential Evolution for global exploration
        # DE with population size ~ 10*n, max iterations limited by budget
        pop_size = max(10, 5 * n)
        max_iter_de = min(50, (budget - 10) // pop_size)
        
        if max_iter_de > 0:
            bounds = [(-5.0, 5.0)] * n
            # Use a moderate number of initial points
            init_points = min(pop_size, 50)
            
            def de_callback(xk, convergence):
                nonlocal best_x, best_f
                return False  # Let it run to max_iter or timeout

            try:
                result_de = differential_evolution(
                    timed_func,
                    bounds,
                    maxiter=max_iter_de,
                    popsize=pop_size,
                    tol=1e-10,
                    atol=1e-12,
                    seed=12345,
                    init='latinhypercube',
                    polish=False,  # We do our own polish
                    updateinterval=1,
                    callback=de_callback
                )
                if result_de.success or result_de.fun < best_f:
                    best_f = result_de.fun
                    best_x = result_de.x
            except Exception:
                pass

        # Phase 2: Multi-start local refinement with BFGS/L-BFGS-B
        # Use remaining budget for local optimizations from multiple starts
        remaining_budget = budget - call_count
        if remaining_budget > 10 and best_x is not None:
            # Start from the best DE point and a few random points
            start_points = [best_x.copy()]
            
            # Add a few random points near the best solution
            for i in range(3):
                perturbation = rng.standard_normal(n) * 0.5
                xp = best_x + perturbation
                xp = np.clip(xp, -5.0, 5.0)
                start_points.append(xp)
            
            # Also add a few global random starts if budget allows
            for i in range(2):
                xr = rng.uniform(-5, 5, n)
                start_points.append(xr)

            for sp in start_points:
                if call_count >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                
                try:
                    # Use L-BFGS-B for box-constrained optimization
                    res = minimize(
                        timed_func,
                        sp,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Phase 3: Final polish with Nelder-Mead if budget remains
        if call_count < budget - 5 and best_x is not None and time.time() < deadline:
            try:
                res = minimize(
                    timed_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Ensure we have a valid best_x
        if best_x is None:
            best_x = np.zeros(n)
            best_f = timed_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
