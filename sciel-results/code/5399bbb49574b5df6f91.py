import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard
        start_time = time.time()
        deadline = start_time + 285

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Diverse initial sampling to find promising regions
        n_initial = min(30, budget)
        initial_points = []

        # Center
        initial_points.append(np.zeros(dim))

        # Axis-aligned points at various distances
        for i in range(dim):
            for dist in [1.0, 2.0, 3.0]:
                p = np.zeros(dim)
                p[i] = dist
                initial_points.append(p)
                p2 = np.zeros(dim)
                p2[i] = -dist
                initial_points.append(p2)

        # Random points
        while len(initial_points) < n_initial:
            initial_points.append(rng.uniform(-4.5, 4.5, size=dim))

        initial_points = initial_points[:n_initial]

        # Evaluate initial points
        for p in initial_points:
            if evals_used >= budget or time.time() > deadline:
                break
            try_func(p)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead simplex method
        remaining = budget - evals_used
        if remaining < 20:
            return best_x, best_f

        def nelder_mead(x0, budget_alloc):
            nonlocal evals_used, best_f, best_x

            def nm_eval(x):
                if evals_used >= budget or time.time() > deadline:
                    return np.inf
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                f_val = func(x)
                evals_used += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val

            # Initialize simplex
            alpha = 0.5
            simplex = [x0.copy()]
            f_vals = [nm_eval(x0)]
            evals = 1

            for i in range(dim):
                if evals >= budget_alloc or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] += alpha if xi[i] < 4.5 else -alpha
                simplex.append(xi.copy())
                f_vals.append(nm_eval(xi))
                evals += 1

            if evals >= budget_alloc or time.time() > deadline:
                return

            # Nelder-Mead parameters
            reflection = 1.0
            expansion = 2.0
            contraction = 0.5
            shrink = 0.5

            while evals < budget_alloc and time.time() < deadline:
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i].copy() for i in order]
                f_vals = [f_vals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + reflection * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = nm_eval(x_ref)
                evals += 1
                if np.isinf(f_ref):
                    break

                if f_vals[0] <= f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                elif f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + expansion * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = nm_eval(x_exp)
                    evals += 1
                    if np.isinf(f_exp):
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_vals[-1]:
                        x_con = centroid + contraction * (x_ref - centroid)
                    else:
                        x_con = centroid - contraction * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = nm_eval(x_con)
                    evals += 1
                    if np.isinf(f_con):
                        break

                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = nm_eval(simplex[i])
                            evals += 1
                            if np.isinf(f_vals[i]) or evals >= budget_alloc or time.time() > deadline:
                                break
                        if evals >= budget_alloc or time.time() > deadline:
                            break

        # Run multiple Nelder-Mead starts
        n_starts = 5
        budget_per_start = remaining // n_starts

        for s in range(n_starts):
            if evals_used >= budget or time.time() > deadline:
                break

            if s == 0:
                x_start = best_x.copy()
            else:
                perturbation = rng.uniform(-0.5, 0.5, size=dim)
                x_start = np.clip(best_x + perturbation, -5.0, 5.0)

            nelder_mead(x_start, budget_per_start)

        return best_x, best_f
