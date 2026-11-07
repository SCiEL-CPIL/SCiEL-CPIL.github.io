"""Nelder-Mead simplex with adaptive multi-restart and coordinate-descent polishing for BBOB f20 (Schwefel)."""
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead simplex optimizer with coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() >= deadline:
                return np.inf
            calls_used += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        def nelder_mead(func_eval, x0, initial_scale, max_iter):
            """Run Nelder-Mead from x0 with given initial simplex scale and max iterations."""
            dim_local = len(x0)
            simplex = [x0.copy()]
            fvals = [func_eval(x0)]

            for i in range(dim_local):
                xi = x0.copy()
                xi[i] += initial_scale
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                fvals.append(func_eval(xi))

            if calls_used >= budget or time.time() >= deadline:
                idx = np.argmin(fvals)
                return np.array(simplex[idx]), fvals[idx]

            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if calls_used >= budget or time.time() >= deadline:
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func_eval(xr)

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func_eval(xc)
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim_local + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func_eval(simplex[i])

                spread = max(fvals) - min(fvals)
                if spread < 1e-10:
                    break

            idx_min = np.argmin(fvals)
            return np.array(simplex[idx_min]), fvals[idx_min]

        def coordinate_descent(func_eval, x0, max_iter, step_size=0.5):
            """Simple coordinate descent for final polishing."""
            x = x0.copy()
            f = func_eval(x)
            for _ in range(max_iter):
                if calls_used >= budget or time.time() >= deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        xn = x.copy()
                        xn[d] += direction * step_size
                        xn = np.clip(xn, -5.0, 5.0)
                        fn = func_eval(xn)
                        if fn < f:
                            x = xn
                            f = fn
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
            return x, f

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Generate diverse starting points
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        # Axis-aligned points at ±4 (probe Schwefel periodic structure)
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 4.0
                starts.append(p)
        # Random points
        for _ in range(6):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Corner-ish points
        for _ in range(2):
            starts.append(rng.choice([-4, 0, 4], size=dim).astype(float))

        # Shuffle to mix
        rng.shuffle(starts)

        # Phase 1: Coarse exploration - try several starts with large scale
        # Use ~30% of budget
        phase1_budget = int(budget * 0.30)
        n_coarse = min(3, len(starts))
        calls_per_coarse = phase1_budget // n_coarse

        for i in range(n_coarse):
            if calls_used >= budget or time.time() >= deadline:
                break
            x_res, f_res = nelder_mead(safe_func, starts[i], 3.0, calls_per_coarse)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 2: Medium refinement from best found
        if best_x is not None and calls_used < budget and time.time() < deadline:
            phase2_budget = int(budget * 0.25)
            x_res2, f_res2 = nelder_mead(safe_func, best_x, 1.0, phase2_budget)
            if f_res2 < best_f:
                best_f = f_res2
                best_x = x_res2.copy()

        # Phase 3: Fine refinement
        if best_x is not None and calls_used < budget and time.time() < deadline:
            phase3_budget = int(budget * 0.20)
            x_res3, f_res3 = nelder_mead(safe_func, best_x, 0.3, phase3_budget)
            if f_res3 < best_f:
                best_f = f_res3
                best_x = x_res3.copy()

        # Phase 4: Coordinate descent polishing
        if best_x is not None and calls_used < budget - 5 and time.time() < deadline:
            remaining = budget - calls_used
            max_cd_iter = min(remaining // 10, 50)
            x_res4, f_res4 = coordinate_descent(safe_func, best_x, max_cd_iter, step_size=0.3)
            if f_res4 < best_f:
                best_f = f_res4
                best_x = x_res4.copy()

        # Phase 5: Final ultra-fine NM if budget remains
        if best_x is not None and calls_used < budget - 10 and time.time() < deadline:
            remaining = budget - calls_used
            if remaining > 15:
                x_res5, f_res5 = nelder_mead(safe_func, best_x, 0.1, remaining - 5)
                if f_res5 < best_f:
                    best_f = f_res5
                    best_x = x_res5.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
