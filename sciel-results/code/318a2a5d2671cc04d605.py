"""Improved BBOB f22 optimizer: Multi-Start Nelder-Mead with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with LHS exploration and final polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead(x_start, f_start, max_steps=50, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        # Phase 1: LHS Exploration (25 points)
        n_lhs = 25
        points = []
        if calls_made + n_lhs <= self.budget:
            lhs_points = np.zeros((n_lhs, n))
            for i in range(n):
                perm = rng.permutation(n_lhs)
                u = (perm + 0.5) / n_lhs
                lhs_points[:, i] = -5 + 10 * u
            for x in lhs_points:
                if not check_time() or calls_made >= self.budget:
                    break
                fx = safe_func(x)
                points.append((fx, x.copy()))

        # Add center and a few random points for diversity
        extra_starts = [
            np.zeros(n),
            rng.uniform(-3, 3, size=n),
            rng.uniform(-3, 3, size=n),
        ]
        for x in extra_starts:
            if not check_time() or calls_made >= self.budget:
                break
            fx = safe_func(x)
            points.append((fx, x.copy()))

        points.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Nelder-Mead on top candidates
        top_k = 6
        remaining = self.budget - calls_made
        nm_budget_per_start = max(30, remaining // (top_k + 2))

        for i in range(min(top_k, len(points))):
            if not check_time() or calls_made >= self.budget:
                break
            x_start = points[i][1]
            f_start = points[i][0]
            # Adaptive delta: larger for top candidate, smaller for others
            delta = 0.5 if i == 0 else 0.3
            nelder_mead(x_start.copy(), f_start, max_steps=nm_budget_per_start, initial_delta=delta)

        # Phase 3: Final fine polish on the best point
        if check_time() and calls_made < self.budget and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=25, initial_delta=0.1)

        return best_x, best_f

# EVOLVE-BLOCK-END
