"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive refinement and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(1234)

        t_start = time.time()
        deadline = 290.0  # safety margin below the 300s wall-clock limit

        # Count function evaluations against the budget.
        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        # ---- Phase 1: Nelder-Mead over the whole box ----
        # Start at the center; the hidden minimizer lies in [-4,4]^5, so the
        # center is a reasonable, robust initial point.
        x = np.zeros(dim)
        fx = f(x)
        best_x = x.copy()
        best_fx = fx

        # Build an initial simplex with a generous spread so the search
        # quickly spans the box.
        simplex = [x.copy()]
        for i in range(dim):
            v = x.copy()
            v[i] = x[i] + 4.0
            simplex.append(np.clip(v, -5.0, 5.0))
        fvals = [f(s) for s in simplex]

        # Track best.
        for s, fv in zip(simplex, fvals):
            if fv < best_fx:
                best_fx, best_x = fv, s.copy()

        alpha, gamma, rho, sigma = 1.0, 2.0, -1.0, 0.5

        def update_best(s, fv):
            nonlocal best_fx, best_x
            if fv < best_fx:
                best_fx, best_x = fv, s.copy()

        # Nelder-Mead iterations, bounded by remaining budget and deadline.
        while calls[0] < budget and (time.time() - t_start) < deadline:
            # Sort simplex by function value.
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            xw = simplex[-1]     # worst
            xg = simplex[1]      # second best (for reflection center)
            xc = np.mean(simplex[:-1], axis=0)  # centroid of all but worst
            fw = fvals[-1]
            fb = fvals[0]

            if calls[0] >= budget:
                break

            # Reflection.
            xr = np.clip(xc + alpha * (xc - xw), -5.0, 5.0)
            fr = f(xr)
            update_best(xr, fr)

            if fb <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fb:
                # Expansion.
                if calls[0] >= budget:
                    break
                xe = np.clip(xc + gamma * (xc - xw), -5.0, 5.0)
                fe = f(xe)
                update_best(xe, fe)
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                # Contraction.
                if fr < fw:
                    xs = np.clip(xc + rho * (xw - xc), -5.0, 5.0)
                else:
                    xs = np.clip(xc + sigma * (xw - xc), -5.0, 5.0)
                if calls[0] >= budget:
                    break
                fs = f(xs)
                update_best(xs, fs)
                if fs < min(fw, fr):
                    simplex[-1] = xs
                    fvals[-1] = fs
                else:
                    # Shrink.
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(xb if False else simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                        if calls[0] >= budget:
                            break
                        fvals[i] = f(simplex[i])
                        update_best(simplex[i], fvals[i])
                    if calls[0] >= budget:
                        break

        # ---- Phase 2: local (1+1)-ES polish from the best point ----
        x = best_x.copy()
        fx = best_fx
        sigma = 0.3
        while calls[0] < budget and (time.time() - t_start) < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = f(y)
            if fy < fx:
                x, fx = y, fy
                best_x, best_fx = x.copy(), fy
                sigma *= 1.3
            else:
                sigma *= 0.85
                if sigma < 1e-4:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
