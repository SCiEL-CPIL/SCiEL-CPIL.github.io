"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with (1+1)-ES polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def nelder_mead(x0, fx0, max_iter=80):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if x0[i] < 4.9 else -0.1
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    return
                simplex.append(xi)
                fvals.append(fi)

            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(max_iter):
                if time.time() > deadline - 10.0 or calls_made >= self.budget:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                x_centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]

                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)
                if f_ref is None:
                    break

                if f_ref < fvals[0]:
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_call(x_exp)
                    if f_exp is None:
                        break
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
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)
                    if f_con is None:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_call(simplex[i])
                            if fvals[i] is None:
                                return

        def polish(x, fx, sigma, max_iters=200):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            for _ in range(max_iters):
                if time.time() > deadline - 5.0 or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break
            return x, fx

        # Phase 1: Multi-start Nelder-Mead
        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(self.dim))
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = 3.0
            starts.append(pt)
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = -3.0
            starts.append(pt)
        while len(starts) < 10:
            starts.append(rng.uniform(-4, 4, self.dim))

        # Sort starts by initial function value (evaluate each)
        start_fvals = []
        for x_s in starts:
            if time.time() > deadline - 30.0 or calls_made >= self.budget:
                break
            fx_s = safe_call(x_s)
            if fx_s is None:
                break
            start_fvals.append((fx_s, x_s))

        # Sort by function value
        start_fvals.sort(key=lambda t: t[0])

        # Run Nelder-Mead on the best starts, allocating budget proportionally
        n_nm_starts = min(5, len(start_fvals))
        for i in range(n_nm_starts):
            if time.time() > deadline - 20.0 or calls_made >= self.budget:
                break
            fx_s, x_s = start_fvals[i]
            # Allocate more iterations to better starts
            max_iter = max(30, 80 - 10 * i)
            nelder_mead(x_s, fx_s, max_iter=max_iter)

        # Phase 2: (1+1)-ES polish from the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            x, fx = polish(x, fx, sigma, max_iters=150)

        # Phase 3: If budget remains, do another polish with smaller sigma
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 3.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            x, fx = polish(x, fx, sigma, max_iters=100)

        return best_x, best_f
# EVOLVE-BLOCK-END
