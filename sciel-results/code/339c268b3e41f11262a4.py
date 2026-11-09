"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
  1. Brief (1+1)-ES with aggressive sigma adaptation for global convergence.
  2. Nelder-Mead local polish to exploit the smooth, unimodal geometry of f14.
  3. If NM does not improve, a final short (1+1)-ES refinement with small sigma.

The hidden affine transform (shift + rotation + offset) preserves the unimodal,
smooth nature of f14, so local methods like NM are effective once the global
search has placed us near the basin of attraction.
"""
import time
import numpy as np


class Optimizer:
    """(1+1)-ES + Nelder-Mead polish optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 310.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # ---- Phase 0: Initial evaluation at origin ----
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        # ---- Phase 1: (1+1)-ES global search ----
        # Use a moderate number of restarts to cover the search space,
        # then commit remaining budget to local polish.
        def run_es(x_init, f_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            x = x_init.copy()
            f = f_init
            sigma = sigma_init
            no_improve = 0
            stagnation_limit = 12 * dim

            while still_ok() and calls < budget and calls < max_calls:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < f:
                    x = y.copy()
                    f = fy
                    no_improve = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.08  # aggressive increase on success
                else:
                    no_improve += 1
                    sigma *= 0.90  # faster decrease on failure

                sigma = np.clip(sigma, 1e-9, 10.0)

                if no_improve >= stagnation_limit:
                    break

            return x, f, sigma

        # Allocate ~40% of budget to global ES, rest to local polish
        es_budget = max(50, int(budget * 0.40))
        n_restarts = 3
        calls_per_restart = max(10, es_budget // n_restarts)

        # First ES run from current best
        x, f, sigma = run_es(best_x, best_f, 2.0, calls + calls_per_restart)

        # Additional restarts from random points
        for r in range(1, n_restarts):
            if not still_ok() or calls >= budget:
                break
            x_init = rng.uniform(-5.0, 5.0, dim)
            f_init = make_call(x_init)
            if f_init is None:
                break

            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

            x, f, sigma = run_es(x_init, f_init, 2.0, calls + calls_per_restart)

        # ---- Phase 2: Nelder-Mead local polish ----
        # Build initial simplex around best_x
        if still_ok() and calls < budget:
            x_best = best_x.copy()
            f_best = best_f

            # Simplex: best_x + small perturbations along each axis
            simplex_size = 0.5
            simplex = [x_best.copy()]
            for i in range(dim):
                vertex = x_best.copy()
                vertex[i] += simplex_size
                simplex.append(np.clip(vertex, -5.0, 5.0))

            # Evaluate simplex vertices
            simplex_f = []
            for v in simplex:
                fv = make_call(v)
                if fv is None:
                    break
                simplex_f.append(fv)
                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()
                    best_f = fv
                    best_x = v.copy()
            else:
                # All simplex vertices evaluated; run Nelder-Mead
                simplex_f = np.array(simplex_f)
                simplex = np.array(simplex)

                # Standard Nelder-Mead parameters
                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma_nm = 0.5  # shrink

                nm_max_iter = 200
                nm_iter = 0

                while (nm_iter < nm_max_iter and still_ok() and calls < budget):
                    nm_iter += 1
                    # Sort by function value
                    order = np.argsort(simplex_f)
                    simplex = simplex[order]
                    simplex_f = simplex_f[order]

                    # Check convergence
                    if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                        break

                    x_worst = simplex[-1]
                    f_worst = simplex_f[-1]
                    x_best_nm = simplex[0]
                    f_best_nm = simplex_f[0]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + alpha * (centroid - x_worst)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = make_call(x_ref)
                    if f_ref is None:
                        break

                    if f_ref < f_best_nm:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = make_call(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            x_new, f_new = x_exp, f_exp
                        else:
                            x_new, f_new = x_ref, f_ref
                        simplex[-1] = x_new
                        simplex_f[-1] = f_new
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (x_worst - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = make_call(x_con)
                        if f_con is None:
                            break
                        if f_con < f_worst:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best_nm + sigma_nm * (simplex[i] - x_best_nm)
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_sh = make_call(simplex[i])
                                if f_sh is None:
                                    break
                                simplex_f[i] = f_sh
                            else:
                                pass
                            break

                    # Update best
                    for i in range(dim + 1):
                        if simplex_f[i] < f_best:
                            f_best = simplex_f[i]
                            x_best = simplex[i].copy()
                            best_f = f_best
                            best_x = x_best.copy()

        # ---- Phase 3: Final short (1+1)-ES refinement if budget remains ----
        if still_ok() and calls < budget:
            x, f, sigma = run_es(best_x, best_f, 0.1, budget)

        return best_x, float(best_f)
