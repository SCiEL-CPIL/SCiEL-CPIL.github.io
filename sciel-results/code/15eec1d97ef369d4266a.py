"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 285.0  # Safety margin for 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        # Phase 1: Multi-start exploration with (1+1)-ES
        # Rosenbrock has a narrow valley; diverse starts increase chance of finding it
        num_starts = 6
        es_budget_per_start = max(30, min(70, (budget - 120) // num_starts))

        # Structured + random starting points
        starts = []
        # Points along the Rosenbrock valley direction (1,1,...,1)/sqrt(d)
        valley_dir = np.ones(dim) / np.sqrt(dim)
        for offset in [0.0, 1.5, -1.5, 2.5]:
            starts.append(offset * valley_dir + rng.uniform(-0.5, 0.5, dim))
        # Two additional random points for diversity
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))

        for s_idx, x0 in enumerate(starts):
            if not check_time() or calls_made >= budget - 60:
                break

            x = np.clip(x0, -5, 5)
            try:
                fx = safe_call(x)
            except StopIteration:
                break

            # (1+1)-ES with adaptive sigma
            sigma = 1.5
            successes = 0
            for it in range(es_budget_per_start):
                if not check_time() or calls_made >= budget - 60:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = safe_call(y)
                except StopIteration:
                    break
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    # Increase sigma on success, but cap it
                    sigma = min(sigma * 1.15, 4.0)
                else:
                    successes = 0
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.85, 0.05)

        # Phase 2: Local polish with L-BFGS-B from the best point found
        if check_time() and calls_made < budget - 10 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 200, 'maxfun': budget - calls_made})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: If budget remains and we're not at a very good value,
        # try a second polish from a perturbed best point
        if check_time() and calls_made < budget - 10 and best_f > 0.5 and best_x is not None:
            x2 = np.clip(best_x + 0.3 * rng.standard_normal(dim), -5, 5)
            try:
                f2 = safe_call(x2)

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                res2 = minimize(lbfun2, x2, method='L-BFGS-B',
                                bounds=[(-5.0, 5.0) for _ in range(dim)],
                                options={'maxiter': 100, 'maxfun': budget - calls_made})
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Use any remaining calls for a final short random search
        # around the best point to potentially escape a local minimum
        if check_time() and calls_made < budget - 5 and best_x is not None:
            for _ in range(min(50, budget - calls_made)):
                if not check_time() or calls_made >= budget:
                    break
                x3 = np.clip(best_x + 0.2 * rng.standard_normal(dim), -5, 5)
                try:
                    safe_call(x3)
                except StopIteration:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
