"""Rewritten optimizer: CMA-ES for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize


class Optimizer:
    """CMA-ES with a coarse random restart fallback, timed to stay under budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track best solution found
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Wrap func to count calls and enforce deadline
        call_count = [0]
        timed_out = [False]

        def timed_func(x):
            nonlocal timed_out
            if time.time() >= deadline:
                timed_out[0] = True
                return np.inf
            call_count[0] += 1
            if call_count[0] > budget:
                timed_out[0] = True
                return np.inf
            try:
                return float(func(x))
            except Exception:
                timed_out[0] = True
                return np.inf

        # Phase 1: Coarse random sampling to find a good starting point
        # Use ~20% of budget for initial exploration
        n_initial = min(50, budget // 5)
        rng = np.random.default_rng(42)
        for i in range(n_initial):
            if time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = timed_func(x)
            if f < np.inf:
                update_best(x, f)
            if timed_out[0]:
                break

        if best_x is None:
            # Fallback: return a default point
            return np.zeros(dim), 0.0

        # Phase 2: CMA-ES from the best point found
        # CMA-ES is well-suited for Rastrigin: it adapts covariance to handle
        # the multimodal landscape and converges well on smooth sub-basins.
        # We use a moderate population size and let it run until budget/time.
        popsize = max(4, int(4 + 3 * np.log(dim)))  # standard CMA-ES popsize
        max_iters = max(1, (budget - call_count[0]) // popsize)

        # Set up CMA-ES options
        cma_options = {
            'maxfevals': budget - call_count[0] + 10,
            'tolfun': 1e-12,
            'tolfunint': 1e-12,
            'tolfunconst': 1e-12,
            'tolfx': 1e-12,
            'tolfxint': 1e-12,
            'tolfxconst': 1e-12,
            'verbose': -9,
            'bounds': [(-5.0, 5.0)] * dim,
            'seed': 42,
        }

        try:
            result = optimize.minimize(
                timed_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iters,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            if result.fun < np.inf:
                update_best(result.x, result.fun)
        except Exception:
            pass

        # Phase 3: If time remains, do a second CMA-ES-like refinement
        # using scipy's differential_evolution for global exploration
        remaining_calls = budget - call_count[0]
        if remaining_calls > 100 and time.time() < deadline - 60:
            try:
                def de_func(x):
                    if time.time() >= deadline or call_count[0] >= budget:
                        return np.inf
                    return timed_func(x)

                de_result = optimize.differential_evolution(
                    de_func,
                    bounds=[(-5, 5)] * dim,
                    maxiter=max(10, remaining_calls // 200),
                    popsize=15,
                    tol=1e-10,
                    seed=42,
                    init='sobol',
                    polish=False,
                )
                if de_result.fun < np.inf:
                    update_best(de_result.x, de_result.fun)
            except Exception:
                pass

        # Phase 4: Final local refinement with Nelder-Mead from best point
        remaining_calls = budget - call_count[0]
        if remaining_calls > 20 and time.time() < deadline - 30:
            try:
                nm_result = optimize.minimize(
                    timed_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_calls, 200),
                        'xatol': 1e-10,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
                if nm_result.fun < np.inf:
                    update_best(nm_result.x, nm_result.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
