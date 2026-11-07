"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with a final CMA-ES-like polish, budget-aware."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Count function calls
        call_count = [0]
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Multi-start L-BFGS-B
        # Use a fraction of budget for initial exploration
        n_starts = 15
        budget_per_start = max(10, (budget - 50) // n_starts)

        # Initial points: a mix of random and structured
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for i, x0 in enumerate(starts):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            x0 = np.clip(x0, -5, 5)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': budget_per_start,
                             'maxfun': budget_per_start,
                             'gtol': 1e-6,
                             'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 2: Refinement around best point with smaller steps
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 5:
                # Try a few L-BFGS-B refinements from perturbed best
                n_refine = min(5, remaining // 5)
                for _ in range(n_refine):
                    if call_count[0] >= budget:
                        break
                    if time.time() > deadline:
                        break
                    perturb = best_x + rng.normal(0, 0.1, dim)
                    perturb = np.clip(perturb, -5, 5)
                    try:
                        res = minimize(
                            wrapped_func,
                            perturb,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 20,
                                     'maxfun': 20,
                                     'gtol': 1e-8,
                                     'ftol': 1e-12}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Phase 3: Final polish with very tight tolerance
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 3:
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50,
                                 'maxfun': remaining,
                                 'gtol': 1e-10,
                                 'ftol': 1e-14}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
