"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex ensemble for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def nelder_mead(x0, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """Nelder-Mead simplex method. Returns (best_x, best_f)."""
            nonlocal call_count, best_x, best_f
            dim = self.dim
            n = dim + 1
            step = 0.5

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                pt = x0.copy()
                pt[i] += step
                simplex.append(pt)

            # Evaluate initial simplex
            fvals = []
            for pt in simplex:
                if call_count >= budget or time.time() > deadline:
                    return x0, np.inf
                pt = np.clip(pt, -5, 5)
                fv = call_func(pt)
                if fv is None:
                    return x0, np.inf
                fvals.append(fv)

            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if call_count >= budget or time.time() > deadline:
                    break
                fr = call_func(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if call_count >= budget or time.time() > deadline:
                        break
                    fe = call_func(xe)
                    if fe is None:
                        break
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    if call_count >= budget or time.time() > deadline:
                        break
                    fc = call_func(xc)
                    if fc is None:
                        break
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if call_count >= budget or time.time() > deadline:
                                break
                            fvals[i] = call_func(simplex[i])
                            if fvals[i] is None:
                                return x0, np.inf
                        if fvals[1] is None:
                            break

                # Track global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Phase 1: Multi-start Nelder-Mead ensemble
        # Initialize 5 simplexes at different locations
        # Corners of the search space and center
        init_points = [
            np.array([0.0] * dim),  # center
            np.array([4.0] * dim),  # corner
            np.array([-4.0] * dim),  # opposite corner
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),  # mixed corner
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),  # mixed corner
        ]

        # Evaluate initial points
        for pt in init_points:
            if call_count >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            fv = call_func(pt)
            if fv is None:
                break

        # Run Nelder-Mead from each starting point
        for i, pt in enumerate(init_points):
            if call_count >= budget or time.time() > deadline:
                break
            # Use remaining budget divided among remaining simplexes
            remaining = budget - call_count
            max_iter = max(50, remaining // (len(init_points) - i))
            nelder_mead(pt, max_iter=max_iter)

        # Phase 2: If we have budget left, do additional local refinement
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            # Fine coordinate descent
            for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                if call_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break

        # Phase 3: Random restarts if we haven't found a good solution
        if call_count < budget and time.time() < deadline and best_f > 5.0:
            n_restarts = min(10, (budget - call_count) // 20)
            for _ in range(n_restarts):
                if call_count >= budget or time.time() > deadline:
                    break
                # Random point near integer grid
                pt = np.array([rng.choice(range(-4, 5)) for _ in range(dim)], dtype=float)
                pt += rng.uniform(-0.3, 0.3, dim)
                pt = np.clip(pt, -5, 5)

                fv = call_func(pt)
                if fv is None:
                    break

                if fv < best_f:
                    # Local refinement
                    x = pt.copy()
                    fx = fv
                    for step in [0.1, 0.05, 0.02, 0.01]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        improved = True
                        while improved and call_count < budget and time.time() < deadline:
                            improved = False
                            for d in range(dim):
                                if call_count >= budget or time.time() > deadline:
                                    break
                                for direction in [1, -1]:
                                    x_new = x.copy()
                                    x_new[d] += direction * step
                                    x_new = np.clip(x_new, -5, 5)
                                    fx_new = call_func(x_new)
                                    if fx_new is None:
                                        break
                                    if fx_new < fx - 1e-12:
                                        x = x_new
                                        fx = fx_new
                                        improved = True
                                        if fx < best_f:
                                            best_f = fx
                                            best_x = x.copy()
                                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
