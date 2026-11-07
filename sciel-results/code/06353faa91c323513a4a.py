"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with restarts for Schwefel-like f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            calls_used += 1
            return x_clipped, f

        def nelder_mead(x0, f0, max_iters, sigma_init=0.5):
            nonlocal best_x, best_f, calls_used
            dim = self.dim

            # Build simplex
            simplex = [x0.copy()]
            fvals = [f0]

            for i in range(dim):
                xi = x0.copy()
                xi[i] += sigma_init
                xi = np.clip(xi, -5.0, 5.0)
                res = evaluate(xi)
                if res is None:
                    return
                simplex.append(res[0])
                fvals.append(res[1])
                if res[1] < best_f:
                    best_f = res[1]
                    best_x = res[0].copy()

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iters):
                if time.time() > deadline:
                    return
                if calls_used >= budget:
                    return

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break

                x_best = simplex[0]
                x_worst = simplex[-1]
                x_second_worst = simplex[-2]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                res = evaluate(x_ref)
                if res is None:
                    return
                x_ref, f_ref = res
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    res = evaluate(x_exp)
                    if res is None:
                        return
                    x_exp, f_exp = res
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + sigma * (x_worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    res = evaluate(x_con)
                    if res is None:
                        return
                    x_con, f_con = res
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()

                    if f_con < f_ref:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            res = evaluate(simplex[i])
                            if res is None:
                                return
                            simplex[i], fvals[i] = res
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

            # After NM, do a local refinement
            if best_x is not None and calls_used < budget and time.time() < deadline:
                self._local_refine(func, best_x, best_f)

        def _local_refine(self, func, x_best, f_best):
            nonlocal calls_used, best_x, best_f
            dim = self.dim
            sigma = 0.1
            for _ in range(30):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(dim):
                    for direction in [-1, 1]:
                        x_try = x_best.copy()
                        x_try[i] += direction * sigma
                        x_try = np.clip(x_try, -5.0, 5.0)
                        res = evaluate(x_try)
                        if res is None:
                            return
                        x_try, f_try = res
                        if f_try < f_best - 1e-15:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            if f_best < best_f:
                                best_f = f_best
                                best_x = x_best.copy()
                            break
                    if improved:
                        break
                if not improved:
                    sigma *= 0.5
                    if sigma < 1e-8:
                        break

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # Phase 1: diverse initial points
        n_starts = 8
        for s in range(n_starts):
            if time.time() > deadline or calls_used >= budget:
                break

            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)

            res = evaluate(x0)
            if res is None:
                break
            x0, f0 = res
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run Nelder-Mead with limited iterations
            remaining = budget - calls_used
            max_iters = max(5, min(60, remaining // (dim + 1)))
            nelder_mead(x0, f0, max_iters, sigma_init=0.8 if s == 0 else 0.4)

        # Phase 2: if we have budget left, do more starts around best
        if best_x is not None and calls_used < budget and time.time() < deadline:
            n_extra = 4
            for s in range(n_extra):
                if time.time() > deadline or calls_used >= budget:
                    break
                x0 = best_x + rng.normal(0, 0.3, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                res = evaluate(x0)
                if res is None:
                    break
                x0, f0 = res
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

                remaining = budget - calls_used
                max_iters = max(3, min(40, remaining // (dim + 1)))
                nelder_mead(x0, f0, max_iters, sigma_init=0.2)

        # Final local refinement
        if best_x is not None and calls_used < budget and time.time() < deadline:
            self._local_refine(func, best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
