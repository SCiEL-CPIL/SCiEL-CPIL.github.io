import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive refinement for f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, f0, max_iter=30, initial_step=0.3):
            nonlocal calls_made
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            for i in range(1, dim + 1):
                if not time_ok():
                    return x0, f0
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step
                fvals[i] = safe_call(simplex[i])

            for _ in range(max_iter):
                if not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.std(fvals) < 1e-8 * (1.0 + abs(fvals[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def coordinate_descent(x, f, max_pass=6, step=0.5):
            nonlocal calls_made
            for _ in range(max_pass):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sign in (1.0, -1.0):
                        if not time_ok():
                            break
                        cand = x.copy()
                        cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                        fc = safe_call(cand)
                        if fc < f - 1e-12:
                            x = cand
                            f = fc
                            improved = True
                            break
                if not improved:
                    break
            return x, f

        # Phase 1: Coarse grid sampling (25 points)
        levels = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        grid_points = []
        # Add origin
        grid_points.append(np.zeros(dim))
        # Add points along each axis
        for d in range(dim):
            for val in levels:
                if val == 0.0:
                    continue
                p = np.zeros(dim)
                p[d] = val
                grid_points.append(p)
        # Add some diagonal points
        for a in range(1, 3):
            for b in range(1, 3):
                for c in range(1, 3):
                    p = np.zeros(dim)
                    p[0] = levels[a]
                    p[1] = levels[b]
                    p[2] = levels[c]
                    p[3] = levels[(a + b + c) % 5]
                    p[4] = levels[(a + 2 * b + 3 * c) % 5]
                    grid_points.append(p)

        for p in grid_points:
            if not time_ok():
                break
            safe_call(p)

        # Phase 2: Local refinement from best point
        if time_ok() and best_x is not None:
            x, f = nelder_mead(best_x, best_f, max_iter=30, initial_step=0.3)
            best_x, best_f = x, f

        # Phase 3: Try a few more starting points if budget allows
        if time_ok() and best_f > 1e-4:
            # Try perturbing the best point in random directions
            rng = np.random.RandomState(42)
            for _ in range(5):
                if not time_ok():
                    break
                perturbation = rng.randn(dim) * 1.0
                x0 = np.clip(best_x + perturbation, -5.0, 5.0)
                f0 = safe_call(x0)
                if f0 < best_f:
                    x, f = nelder_mead(x0, f0, max_iter=20, initial_step=0.2)
                    best_x, best_f = x, f

        # Phase 4: Final coordinate descent polish
        if time_ok() and best_x is not None:
            best_x, best_f = coordinate_descent(best_x, best_f, max_pass=4, step=0.3)

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
