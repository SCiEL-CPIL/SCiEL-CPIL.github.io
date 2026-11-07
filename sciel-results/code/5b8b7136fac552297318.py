"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell optimizer with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, use 290s for safety

        # Phase 1: Multi-start with random points to find good basins
        # Use a portion of budget for exploration
        n_starts = 20
        # Allocate ~40% of budget to exploration (multi-start)
        explore_budget = int(budget * 0.4)
        # Allocate remaining to refinement
        refine_budget = budget - explore_budget

        # Generate starting points
        rng = np.random.default_rng(123)
        starts = [rng.uniform(-4, 4, dim)]  # First point in [-4,4]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        # Evaluate all starting points
        best_x = None
        best_f = np.inf
        evaluated = 0

        for s in starts:
            if evaluated >= explore_budget:
                break
            if time.time() > deadline - 10:
                break
            try:
                f_val = func(s)
                evaluated += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = s.copy()
            except Exception:
                break

        if best_x is None:
            # Fallback: just evaluate one point
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x.copy()
            evaluated = 1

        # Phase 2: Local refinement using Powell method
        # Powell is derivative-free and works well for smooth-ish functions
        # We'll use the best starting point found in Phase 1

        # For Powell, we need to track function evaluations
        eval_count = [evaluated]
        max_powell_evals = refine_budget

        def wrapper(x):
            if eval_count[0] >= max_powell_evals:
                return np.inf
            if time.time() > deadline - 5:
                return np.inf
            try:
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                eval_count[0] += 1
                return val
            except Exception:
                return np.inf

        # Run Powell from the best starting point
        try:
            result = minimize(
                wrapper,
                best_x,
                method='Powell',
                options={'maxiter': 1000, 'maxfev': max_powell_evals, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5)
        except Exception:
            pass

        # Phase 3: If we still have budget, try a few more local optimizations
        # from other good starting points
        if eval_count[0] < budget and time.time() < deadline - 20:
            # Try Powell from a few other random points near the best
            n_extra = 5
            for i in range(n_extra):
                if eval_count[0] >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                # Generate a point near the best solution
                perturbation = rng.uniform(-1, 1, dim)
                x_start = np.clip(best_x + perturbation, -5, 5)
                
                local_max_evals = min(budget - eval_count[0], 50)
                if local_max_evals <= 0:
                    break
                
                def local_wrapper(x):
                    if eval_count[0] >= budget:
                        return np.inf
                    if time.time() > deadline - 5:
                        return np.inf
                    try:
                        x_clipped = np.clip(x, -5, 5)
                        val = func(x_clipped)
                        eval_count[0] += 1
                        return val
                    except Exception:
                        return np.inf

                try:
                    result = minimize(
                        local_wrapper,
                        x_start,
                        method='Powell',
                        options={'maxiter': 500, 'maxfev': local_max_evals, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        # Ensure we don't exceed budget
        # The wrapper already tracks this, but let's be safe

        return best_x, best_f
# EVOLVE-BLOCK-END
