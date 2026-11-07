"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + SLSQP with global exploration and multi-scale refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=np.float64)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                val = func(x_arr)
                val = float(val)
                if np.isfinite(val):
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        def check_deadline():
            return time.time() < deadline and calls_made < budget

        def run_lbfgsb(x0, maxfun, ftol=1e-14, gtol=1e-10):
            """Run L-BFGS-B and return (fun, x) or None on failure."""
            try:
                result = minimize(
                    safe_func, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 500, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol}
                )
                if np.isfinite(result.fun):
                    return float(result.fun), np.clip(result.x, -5, 5)
            except Exception:
                pass
            return None

        def run_slsqp(x0, maxfun, ftol=1e-12):
            """Run SLSQP and return (fun, x) or None on failure."""
            try:
                result = minimize(
                    safe_func, x0, method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': maxfun, 'ftol': ftol}
                )
                if np.isfinite(result.fun):
                    return float(result.fun), np.clip(result.x, -5, 5)
            except Exception:
                pass
            return None

        def try_improve(x0, maxfun, use_slsqp=False):
            """Try to improve best from x0 using specified method."""
            nonlocal best_f, best_x
            if use_slsqp:
                res = run_slsqp(x0, maxfun)
            else:
                res = run_lbfgsb(x0, maxfun)
            if res is not None:
                f, x = res
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    return True
            return False

        rng = np.random.default_rng(42)

        # Phase 1: Random exploration to find promising regions
        n_random = min(20, max(8, budget // 25))
        candidates = []

        for i in range(n_random):
            if not check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization from best candidates using L-BFGS-B
        n_local = min(6, len(candidates), max(1, (budget - calls_made) // 70))

        for i in range(n_local):
            if not check_deadline() or calls_made >= budget - 50:
                break

            fx, x0 = candidates[i]
            remaining = budget - calls_made - 50
            if remaining < 10:
                break

            try_improve(x0, max(10, remaining), use_slsqp=False)

            if time.time() > deadline - 80:
                break

        # Phase 3: SLSQP refinement from the current best point
        if best_x is not None and check_deadline() and calls_made < budget - 30:
            remaining = budget - calls_made - 30
            if remaining >= 10:
                try_improve(best_x, remaining, use_slsqp=True)

        # Phase 4: Multi-scale perturbation refinement
        if best_x is not None and check_deadline():
            scales = [0.3, 0.15, 0.08, 0.04, 0.02]
            for scale in scales:
                if not check_deadline() or calls_made >= budget - 25:
                    break

                n_perturb = min(2, max(1, (budget - calls_made - 15) // 30))
                for _ in range(n_perturb):
                    if not check_deadline() or calls_made >= budget - 20:
                        break

                    perturbation = rng.normal(0, scale, dim)
                    x_pert = np.clip(best_x + perturbation, -5, 5)
                    fx_pert = safe_func(x_pert)

                    if fx_pert < best_f:
                        best_f = fx_pert
                        best_x = x_pert.copy()

                    # Quick local refinement
                    remaining = budget - calls_made - 15
                    if remaining >= 8:
                        try_improve(x_pert, remaining, use_slsqp=False)

        # Phase 5: Additional random restarts with local optimization
        while check_deadline() and calls_made < budget - 40:
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            remaining = budget - calls_made - 40
            if remaining >= 10:
                try_improve(x, remaining, use_slsqp=False)

        # Phase 6: Final high-precision polish with L-BFGS-B
        if best_x is not None and check_deadline() and calls_made < budget - 15:
            remaining = budget - calls_made - 15
            if remaining >= 5:
                try_improve(best_x, remaining, use_slsqp=False)

        # Phase 7: Ultra-precise SLSQP final polish (quadratic convergence near optimum)
        if best_x is not None and check_deadline() and calls_made < budget - 5:
            remaining = budget - calls_made - 5
            if remaining >= 5:
                try_improve(best_x, remaining, use_slsqp=True)

        # Phase 8: Final L-BFGS-B with tightest tolerances
        if best_x is not None and check_deadline() and calls_made < budget - 2:
            remaining = max(3, budget - calls_made)
            try_improve(best_x, remaining, use_slsqp=False)

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
