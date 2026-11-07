"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) in 5D.
    Strategy:
    1. Sparse global lattice search to identify promising basins (cheap, diverse).
    2. Nelder-Mead simplex method from the best lattice point and a few diverse starts.
       NM is derivative-free, robust to smooth landscapes, and handles the narrow
       Rosenbrock valley well once in the basin.
    3. Adaptive restarts: if NM stagnates, restart from a perturbation of the best
       point or a new random point.
    4. Final (1+1)-ES polish with adaptive step size for the last few calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = np.zeros(d)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x_arr = np.asarray(x, dtype=float)
            fx = float(func(x_arr))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_arr.copy()
            return fx

        # Phase 1: Sparse global lattice search
        # Use a small grid to find diverse starting points.
        # In 5D, a 3^5 = 243 grid would be too many. Use 2^5 = 32 points.
        # Actually, let's use a stratified random sampling for better coverage.
        n_global = min(20, budget // 10)
        global_starts = []
        for _ in range(n_global):
            x = rng.uniform(-4.5, 4.5, d)
            global_starts.append(x)

        # Evaluate all global starts
        for x0 in global_starts:
            if not time_ok() or call_count >= budget - 50:
                break
            fx = tracked_func(x0)

        # Sort global starts by fitness
        global_starts.sort(key=lambda x: tracked_func(x) if call_count < budget else np.inf)
        # Actually we already tracked, so just use best_x and a few others

        # Phase 2: Nelder-Mead from top diverse starts
        # We'll run NM from the best point and a few other diverse points
        nm_starts = [best_x.copy()]
        # Add a few other diverse points from the global search
        for x0 in global_starts[:3]:
            if not np.allclose(x0, best_x, atol=0.5):
                nm_starts.append(x0.copy())

        # Limit NM starts to avoid using too much budget
        nm_starts = nm_starts[:4]

        for i, x0 in enumerate(nm_starts):
            if not time_ok() or call_count >= budget - 30:
                break

            # Run Nelder-Mead with adaptive restarts
            x = x0.copy()
            fx = tracked_func(x) if call_count < budget else np.inf
            if call_count >= budget - 30:
                break

            # Initialize simplex
            sigma = 0.5
            simplex = [x.copy()]
            for j in range(d):
                xj = x.copy()
                xj[j] += sigma
                simplex.append(xj)

            # Evaluate initial simplex
            simplex_f = [tracked_func(s) for s in simplex]
            if call_count >= budget - 30:
                break

            # Nelder-Mead iterations
            max_iter = 100
            for it in range(max_iter):
                if not time_ok() or call_count >= budget - 30:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = tracked_func(xr)
                if call_count >= budget - 30:
                    break

                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = tracked_func(xe)
                    if call_count >= budget - 30:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    fc = tracked_func(xc)
                    if call_count >= budget - 30:
                        break
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, d + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            simplex_f[j] = tracked_func(simplex[j])
                            if call_count >= budget - 30:
                                break
                        if call_count >= budget - 30:
                            break

            # After NM, if we have budget left, try a restart from best
            if time_ok() and call_count < budget - 20 and i < len(nm_starts) - 1:
                # Perturb best and try again
                x_restart = best_x + 0.3 * rng.standard_normal(d)
                x_restart = np.clip(x_restart, -5, 5)
                nm_starts.append(x_restart)

        # Phase 3: (1+1)-ES polish
        if time_ok() and call_count < budget - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            stuck = 0
            remaining = budget - call_count - 5

            for _ in range(remaining):
                if not time_ok() or call_count >= budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    stuck = 0
                else:
                    sigma *= 0.85
                    stuck += 1

                if stuck > 15:
                    x = np.clip(best_x + 0.1 * rng.standard_normal(d), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.05
                    stuck = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
