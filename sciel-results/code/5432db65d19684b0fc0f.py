"""Two-phase optimizer: CMA-ES global exploration + L-BFGS-B local polish for rotated Rosenbrock."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import cma as cma_es


class Optimizer:
    """CMA-ES global search followed by L-BFGS-B local polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 300s kill

        # Counted function wrapper
        call_count = [0]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f[0]:
                best_f[0] = f_val
            return f_val

        # Phase 1: CMA-ES for global exploration
        # CMA-ES adapts its covariance matrix to the landscape anisotropy,
        # which is ideal for the rotated Rosenbrock valley structure.
        # Population size: 4+floor(3*log(5)) = 4+5 = 9 (default)
        # Each generation costs ~pop_size evaluations.
        # Budget allocation: use ~300 calls for CMA-ES (~33 generations)
        # and ~200 calls for L-BFGS-B polish.

        cma_budget = min(budget - 50, 300)  # leave at least 50 for polish
        cma_start = np.zeros(dim)

        try:
            es = cma_es.CMAEvolutionStrategy(
                cma_start,
                2.0,  # initial step size
                {
                    'bounds': [(-5, 5)] * dim,
                    'maxiter': int(cma_budget // 10) + 10,  # approximate generations
                    'popsize': 9,
                    'seed': 42,
                    'verbose': -9,
                    'tolfun': 1e-10,
                    'tolx': 1e-10,
                }
            )

            # Run CMA-ES until budget or convergence
            while call_count[0] < cma_budget and time.time() < deadline:
                if es.should_stop():
                    break
                solutions = es.ask()
                for sol in solutions:
                    if call_count[0] >= cma_budget:
                        break
                    if time.time() > deadline:
                        break
                    sol_clipped = np.clip(sol, -5, 5)
                    f_val = counted_func(sol_clipped)
                    es.tell(sol, f_val)

        except Exception:
            pass

        # Get best point from CMA-ES
        cma_best_x = None
        try:
            cma_best_x = es.result.xbest.copy()
            cma_best_x = np.clip(cma_best_x, -5, 5)
        except Exception:
            cma_best_x = cma_start.copy()

        # Evaluate CMA-ES best if not already evaluated
        f_cma_best = counted_func(cma_best_x)

        # Phase 2: L-BFGS-B local polish from the best point found
        # Use the remaining budget for high-precision local optimization
        polish_budget = budget - call_count[0]

        if polish_budget > 10 and time.time() < deadline:
            try:
                result = minimize(
                    counted_func,
                    cma_best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 500,
                        'maxls': 50,
                        'gtol': 1e-12,
                        'ftol': 1e-15,
                        'maxcor': 20,
                    }
                )
            except Exception:
                result = None

        # If L-BFGS-B didn't help or failed, try from origin as backup
        if result is None or (hasattr(result, 'fun') and result.fun >= best_f[0]):
            if call_count[0] < budget - 10 and time.time() < deadline:
                try:
                    result2 = minimize(
                        counted_func,
                        np.zeros(dim),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 300,
                            'maxls': 50,
                            'gtol': 1e-12,
                            'ftol': 1e-15,
                        }
                    )
                except Exception:
                    result2 = None

        return None, best_f[0]
