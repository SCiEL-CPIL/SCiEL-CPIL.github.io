"""Multi-start L-BFGS-B + Nelder-Mead hybrid for BBOB-style Discus function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead hybrid optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def bounded_lgbfgs(x0, max_evals):
            """Run L-BFGS-B with bounds, tracking evaluations."""
            nonlocal call_count, best_f, best_x

            if call_count >= budget:
                return

            x0 = np.clip(x0, -5.0, 5.0)
            bounds = [(-5.0, 5.0)] * dim
            evals = 0
            maxfunc = max_evals

            def objective(x):
                nonlocal evals
                if evals >= maxfunc or call_count >= budget:
                    return float('inf')
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                evals += 1
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': maxfunc, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                pass

        def nelder_mead(x0, max_evals, initial_simplex=None):
            """Run Nelder-Mead with bounds, tracking evaluations."""
            nonlocal call_count, best_f, best_x

            if call_count >= budget:
                return

            x0 = np.clip(x0, -5.0, 5.0)
            evals = 0
            maxfunc = max_evals

            # Create initial simplex
            if initial_simplex is None:
                simplex = [x0.copy()]
                for i in range(dim):
                    v = x0.copy()
                    step = 0.1 if x0[i] >= 0 else -0.1
                    v[i] += step
                    simplex.append(np.clip(v, -5.0, 5.0))
            else:
                simplex = [np.clip(s, -5.0, 5.0) for s in initial_simplex]

            # Evaluate initial simplex
            for i, s in enumerate(simplex):
                if evals >= maxfunc or call_count >= budget:
                    return
                fx = try_eval(s)
                if fx is None:
                    return
                evals += 1

            while evals < maxfunc and call_count < budget:
                if time.time() > deadline - 5:
                    break

                # Sort simplex by function value
                simplex_f = []
                for s in simplex:
                    # Re-evaluate to get current values (or cache)
                    # For simplicity, we'll track separately
                    pass

                # Standard Nelder-Mead iteration
                # Evaluate all points
                fvals = []
                for s in simplex:
                    if evals >= maxfunc or call_count >= budget:
                        break
                    fx = try_eval(s)
                    if fx is None:
                        return
                    fvals.append(fx)
                    evals += 1

                if len(fvals) < dim + 1:
                    break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = try_eval(x_ref)
                if f_ref is None:
                    break
                evals += 1

                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = try_eval(x_exp)
                    if f_exp is None:
                        break
                    evals += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                    else:
                        simplex[-1] = x_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = try_eval(x_con)
                    if f_con is None:
                        break
                    evals += 1
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fx = try_eval(simplex[i])
                            if fx is None:
                                return
                            evals += 1

        # Phase 1: Multi-start L-BFGS-B from diverse starting points
        n_starts = 15
        start_points = []

        # Origin
        start_points.append(np.zeros(dim))

        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)

        # Random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, size=dim)
            start_points.append(p)

        # Evaluate starting points
        for sp in start_points:
            if call_count >= budget:
                break
            fx = try_eval(sp)
            if fx is None:
                break

        # Run L-BFGS-B from each starting point
        evals_per_lgbfgs = max(20, (budget - call_count) // max(1, len(start_points)))
        for sp in start_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            bounded_lgbfgs(sp, evals_per_lgbfgs)

        # Phase 2: Nelder-Mead refinement from best point
        if call_count < budget and best_x is not None:
            nm_budget = min(100, budget - call_count)
            nelder_mead(best_x.copy(), nm_budget)

        # Phase 3: Final L-BFGS-B polish if budget remains
        if call_count < budget and best_x is not None:
            polish_budget = min(50, budget - call_count)
            if time.time() < deadline - 10:
                bounded_lgbfgs(best_x.copy(), polish_budget)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
