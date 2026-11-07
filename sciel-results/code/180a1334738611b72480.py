"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell local search with a global random-sampling phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        # Phase 1: Global random sampling to find a good starting point
        # Use 50% of budget for exploration
        n_explore = min(self.budget // 2, 250)
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)

        for i in range(n_explore - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Phase 2: Multi-start Powell refinement
        # Use remaining budget for local refinement
        remaining = self.budget - n_explore - 1  # -1 for the initial call already counted
        # Each Powell run uses at most ~2*dim*10 evaluations (heuristic)
        # In 5D, that's about 100 evals per run
        # So we can do at most remaining // 100 Powell runs
        max_powell_runs = max(1, remaining // 150)

        # Generate diverse starting points for Powell
        starts = [x_best.copy()]
        # Add a few more diverse starts from the exploration phase
        for _ in range(min(3, max_powell_runs - 1)):
            starts.append(rng.uniform(-5, 5, self.dim))

        for s_idx, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if remaining <= 0:
                break

            # Run Powell's method
            try:
                result = minimize(
                    lambda x: func(x),
                    x0,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': min(remaining, 200)}
                )
                remaining -= result.nfev

                if result.fun < f_best:
                    f_best = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead if budget remains
        if remaining > 10 and time.time() < deadline:
            try:
                result = minimize(
                    lambda x: func(x),
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < f_best:
                    f_best = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
