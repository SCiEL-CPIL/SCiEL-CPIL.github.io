"""Rewritten optimizer for BBOB f14 (Different powers) using Nelder-Mead multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start initialization
        # Use diverse starting points to cover the search space
        n_starts = 6
        starts = [np.zeros(d)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: Nelder-Mead from each starting point
        # Allocate remaining budget among starts
        remaining_calls = max_calls - calls_used
        calls_per_start = max(10, remaining_calls // (n_starts + 1))

        for start_idx, s in enumerate(starts):
            if calls_used >= max_calls or time.time() > deadline - 10:
                break

            # Initialize simplex around the starting point
            x0 = s.copy()
            fx0 = func(x0)
            calls_used += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Create simplex
            simplex = [x0.copy()]
            simplex_f = [fx0]

            # Add vertices with small perturbations
            for i in range(d):
                xi = x0.copy()
                step = 0.5 if abs(xi[i]) > 0.1 else 0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                if time.time() > deadline - 5:
                    break
                fxi = func(xi)
                calls_used += 1
                if fxi < best_f:
                    best_f = fxi
                    best_x = xi.copy()
                simplex.append(xi.copy())
                simplex_f.append(fxi)

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage

            max_nm_iters = calls_per_start
            for _ in range(max_nm_iters):
                if calls_used >= max_calls or time.time() > deadline - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                calls_used += 1
                if fxr < best_f:
                    best_f = fxr
                    best_x = xr.copy()

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    calls_used += 1
                    if fxe < best_f:
                        best_f = fxe
                        best_x = xe.copy()
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls_used += 1
                    if fxc < best_f:
                        best_f = fxc
                        best_x = xc.copy()

                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrinkage
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx_shrink = func(simplex[i])
                            calls_used += 1
                            if fx_shrink < best_f:
                                best_f = fx_shrink
                                best_x = simplex[i].copy()
                            simplex_f[i] = fx_shrink

        # Phase 3: Final polish with small steps
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_calls = max_calls - calls_used

            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
