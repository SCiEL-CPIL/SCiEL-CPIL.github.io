"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if call_count >= budget:
                return np.inf
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter=200):
            nonlocal call_count, best_f, best_x
            dim = len(x0)
            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            f0 = func(x0)
            simplex.append(f0)

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 0 else -0.5
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                simplex.append(xi)
                simplex.append(fi)

            # Reorganize: list of (f, x) pairs
            verts = [(simplex[2*i], simplex[2*i+1]) for i in range(dim+1)]
            verts.sort(key=lambda v: v[0])

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            no_improve = 0
            max_no_improve = 20

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Sort vertices
                verts.sort(key=lambda v: v[0])

                # Centroid of all but worst
                centroid = np.mean([v[1] for v in verts[:-1]], axis=0)

                # Reflect
                x_worst = verts[-1][1]
                f_worst = verts[-1][0]
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)

                if f_r < verts[0][0]:
                    # Expand
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        verts[-1] = (f_e, x_e)
                    else:
                        verts[-1] = (f_r, x_r)
                    no_improve = 0
                elif f_r < verts[-2][0]:
                    verts[-1] = (f_r, x_r)
                    no_improve = 0
                else:
                    # Contract
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < min(f_r, f_worst):
                        verts[-1] = (f_c, x_c)
                        no_improve = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            verts[i][1] = verts[0][1] + sigma * (verts[i][1] - verts[0][1])
                            verts[i][1] = np.clip(verts[i][1], -5, 5)
                            verts[i][0] = func(verts[i][1])
                        no_improve += 1

                if no_improve >= max_no_improve:
                    break

            return verts[0][1], verts[0][0]

        # Phase 1: Coarse grid search to find promising basins
        # Use a small grid in 5D
        n_per_dim = 3  # 3^5 = 243 points, but we'll use fewer
        # Actually, let's do a smarter initial exploration
        # Start with center and a few random points

        # Initial points: center + random
        initial_points = [np.zeros(dim)]
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, dim))

        # Evaluate initial points
        candidates = []
        for pt in initial_points:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_func(pt)
            candidates.append((f, pt))

        # Sort by fitness
        candidates.sort(key=lambda x: x[0])

        # Phase 2: Run Nelder-Mead from top candidates
        nm_starts = min(5, len(candidates))
        for i in range(nm_starts):
            if time.time() > deadline or call_count >= budget:
                break
            f_start, x_start = candidates[i]
            # Run NM
            x_nm, f_nm = nelder_mead(safe_func, x_start, max_iter=100)

        # Phase 3: Final polish with (1+1)-ES from best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            # Adaptive sigma
            successes = 0
            total = 0
            for _ in range(budget - call_count):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                total += 1
                if total > 0 and total % 10 == 0:
                    # Adjust sigma based on success rate
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
