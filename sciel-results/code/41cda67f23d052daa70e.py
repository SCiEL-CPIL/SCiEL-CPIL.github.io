"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Extensive Sobol sampling to find the narrow valley.
    - Phase 2: Multi-start L-BFGS-B local refinement from top candidates.
    - Phase 3: Final polishing of the global best.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit. We stop at 280s to be safe.
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        # For Discus, we need good coverage to find the valley.
        # 20% of budget for initial exploration is reasonable.
        n_initial = min(100, self.budget // 5)
        if n_initial < 10:
            n_initial = 10

        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50: # Leave some for local search
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            # Fallback
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])

        # Phase 2: Local refinement from the best few points
        # L-BFGS-B is very effective on the smooth Discus function once in the valley.
        remaining_budget = self.budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Select top candidates for local refinement
        # We want to try a few different basins.
        n_restarts = min(10, len(evaluations))
        
        bounds = [(-5, 5)] * self.dim

        # Distribute remaining budget among restarts
        # L-BFGS-B usually converges in < 20 iterations for 5D.
        # We give each restart a generous but capped budget.
        max_iter_per_restart = max(20, remaining_budget // (n_restarts + 1))

        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_used >= self.budget - 10:
                break
                
            fx0, x0 = evaluations[i]
            
            def lbfgs_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_per_restart,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                        'maxls': 50 # Max line search iterations
                    }
                )
            except Exception:
                pass

        # Phase 3: Final polishing
        # If we have budget left, do a very tight optimization from the global best
        if best_x is not None and calls_used < self.budget - 5 and time.time() < deadline - 5:
            remaining = self.budget - calls_used
            if remaining > 10:
                def final_func(x):
                    return safe_call(x)
                
                try:
                    res = minimize(
                        final_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': min(50, remaining),
                            'gtol': 1e-12,
                            'ftol': 1e-15,
                            'maxls': 100
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
