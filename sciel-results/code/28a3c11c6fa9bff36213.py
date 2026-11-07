"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent + Nelder-Mead hybrid for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def coordinate_descent(x0, f0, max_sweeps=100, step_init=1.0):
            """Cyclic coordinate descent with adaptive step sizes and line search."""
            nonlocal calls, best_x, best_f

            x = x0.copy()
            f = f0
            steps = np.full(n, step_init)

            for sweep in range(max_sweeps):
                if time.time() > deadline or calls >= budget:
                    break

                improved = False
                for i in range(n):
                    if time.time() > deadline or calls >= budget:
                        return x, f

                    # Try both directions along coordinate i
                    for direction in [1, -1]:
                        alpha = steps[i]
                        # Backtracking line search
                        for _ in range(10):
                            if calls >= budget or time.time() > deadline:
                                return x, f
                            x_trial = x.copy()
                            x_trial[i] += direction * alpha
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls += 1

                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                steps[i] = min(steps[i] * 1.5, 5.0)
                                improved = True
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                break
                            alpha *= 0.5

                        if calls >= budget or time.time() > deadline:
                            return x, f

                    if not improved:
                        steps[i] *= 0.5

                if not improved:
                    break

            return x, f

        def nelder_mead(x0, f0, max_iter=500):
            """Nelder-Mead simplex method for final polish."""
            nonlocal calls, best_x, best_f

            # Initialize simplex with small perturbations
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.05
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fvals.append(fi)

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[-1] - fvals[0] < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
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
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                return simplex[0], fvals[0]

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        # Phase 1: Evaluate origin and a few random points
        f_origin = func(np.zeros(n))
        calls += 1
        best_x = np.zeros(n).copy()
        best_f = f_origin

        for i in range(8):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Coordinate descent for rapid basin traversal
        x_cd, f_cd = coordinate_descent(best_x.copy(), best_f, max_sweeps=50, step_init=2.0)

        if f_cd < best_f:
            best_f = f_cd
            best_x = x_cd.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Nelder-Mead for exploiting smooth curvature in the valley
        x_nm, f_nm = nelder_mead(best_x.copy(), best_f, max_iter=300)

        if f_nm < best_f:
            best_f = f_nm
            best_x = x_nm.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Additional coordinate descent refinement with smaller steps
        if budget - calls > 20:
            x_cd2, f_cd2 = coordinate_descent(best_x.copy(), best_f, max_sweeps=30, step_init=0.1)
            if f_cd2 < best_f:
                best_f = f_cd2
                best_x = x_cd2.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: Final Nelder-Mead polish
        if budget - calls > 10:
            x_nm2, f_nm2 = nelder_mead(best_x.copy(), best_f, max_iter=100)
            if f_nm2 < best_f:
                best_f = f_nm2
                best_x = x_nm2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
