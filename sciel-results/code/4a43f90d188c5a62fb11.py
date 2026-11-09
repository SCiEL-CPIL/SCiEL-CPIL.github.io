"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with Coordinate Descent Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0

        def call_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            return func(x)

        def nelder_mead(x0, max_iter=100):
            nonlocal calls_used, deadline
            n = self.dim
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.5 if x0[i - 1] >= 0 else -0.5
                simplex[i] = np.clip(simplex[i], -5, 5)

            fvals = [call_func(s) for s in simplex]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = call_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = call_func(simplex[i])
                            break
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = call_func(xc)
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = call_func(simplex[i])
                            break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break

            return simplex[0], fvals[0]

        def coordinate_descent(x, max_iter=100):
            nonlocal calls_used, deadline
            x = x.copy()
            fx = call_func(x)
            step = 1.0

            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(self.dim):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[i] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        if np.all(x_new == x):
                            continue
                        fx_new = call_func(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                        else:
                            step *= 0.5
                    if not improved:
                        step *= 0.5
                if step < 1e-6:
                    break
            return x, fx

        # Initial exploration: Try a few diverse starting points
        best_x = None
        best_f = np.inf

        # Start with origin and a few random points
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.random.default_rng(42).uniform(-4, 4, dim),
            np.random.default_rng(43).uniform(-4, 4, dim)
        ]

        for s in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            s = np.clip(s, -5, 5)
            fs = call_func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Run Nelder-Mead from the best start
        if best_x is not None:
            x_nm, f_nm = nelder_mead(best_x)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()

        # Polish with coordinate descent
        if time.time() < deadline and calls_used < budget:
            x_cd, f_cd = coordinate_descent(best_x)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd.copy()

        # If we have budget left, try NM from another start
        if time.time() < deadline and calls_used < budget:
            for s in starts[1:]:
                if time.time() > deadline or calls_used >= budget:
                    break
                s = np.clip(s, -5, 5)
                # Only if it's different from current best
                if np.linalg.norm(s - best_x) > 1e-3:
                    x_nm2, f_nm2 = nelder_mead(s)
                    if f_nm2 < best_f:
                        best_f = f_nm2
                        best_x = x_nm2.copy()
                        # Polish again
                        if time.time() < deadline and calls_used < budget:
                            x_cd2, f_cd2 = coordinate_descent(best_x)
                            if f_cd2 < best_f:
                                best_f = f_cd2
                                best_x = x_cd2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
