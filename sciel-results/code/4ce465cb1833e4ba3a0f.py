"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for discus-shaped functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return x, fx

        def nelder_mead(x0, fx0, max_iter, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """Nelder-Mead simplex method, returns (best_x, best_f)."""
            nonlocal calls_made, best_x, best_f, deadline
            dim = self.dim
            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [fx0]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if x0[i] >= 0 else -0.1
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                xi, fi = eval_point(xi)
                simplex.append(xi)
                simplex_f.append(fi)

            # Sort by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Reflect
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                xr, fr = eval_point(xr)

                if fr < simplex_f[0]:
                    # Expand
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5.0, 5.0)
                    xe, fe = eval_point(xe)
                    if fe < fr:
                        simplex[-1], simplex_f[-1] = xe, fe
                    else:
                        simplex[-1], simplex_f[-1] = xr, fr
                elif fr < simplex_f[-2]:
                    simplex[-1], simplex_f[-1] = xr, fr
                else:
                    # Contract
                    xc = centroid + sigma * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    xc, fc = eval_point(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i], simplex_f[i] = eval_point(simplex[i])

                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

            return simplex[0].copy(), simplex_f[0]

        # Multi-start strategy
        num_starts = 10
        # First start: center
        x0 = np.zeros(dim)
        x0, fx0 = eval_point(x0)
        best_x, best_f = x0.copy(), fx0

        # Nelder-Mead from center with generous budget
        iter_per_start = max(1, (budget - calls_made - num_starts) // num_starts)

        x0, fx0 = nelder_mead(x0, fx0, iter_per_start)
        if fx0 < best_f:
            best_x, best_f = x0.copy(), fx0

        # Additional starts at random points
        for s in range(num_starts - 1):
            if time.time() > deadline or calls_made >= budget:
                break
            # Random start point
            xr = rng.uniform(-4, 4, dim)
            xr, fxr = eval_point(xr)
            if fxr < best_f:
                best_x, best_f = xr.copy(), fxr
            # Short Nelder-Mead from this point
            remaining = budget - calls_made
            if remaining < 2 * (dim + 1):
                break
            short_iter = max(1, min(iter_per_start // 2, remaining - 2 * (dim + 1)))
            xs, fs = nelder_mead(xr, fxr, short_iter)
            if fs < best_f:
                best_x, best_f = xs.copy(), fs

        return best_x, best_f
# EVOLVE-BLOCK-END
