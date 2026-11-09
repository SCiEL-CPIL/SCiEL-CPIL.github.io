"""Optimizer for BBOB f13 (Sharp ridge) with Nelder-Mead simplex and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with geometrically stretched initial simplex for sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = 0

        def f_count(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Initial point: try a few starting points, pick the best
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Try several random starting points
        n_starts = min(10, budget // 10)
        for i in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = f_count(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = f_count(best_x)

        # Nelder-Mead with geometrically stretched simplex
        # For sharp ridge, stretch along one axis to align with ridge direction
        # We'll use a simplex that is elongated along each coordinate axis

        def nelder_mead(x0, f0, max_iter=200):
            nonlocal call_count
            dim = self.dim

            # Build initial simplex: base point + stretched vertices
            # Stretch factor to align with potential ridge directions
            stretch = 2.0  # geometric stretch for anisotropic landscape

            simplex = [x0.copy()]
            for i in range(dim):
                v = x0.copy()
                v[i] += stretch * 1.0  # unit step stretched
                simplex.append(v)

            # Evaluate initial simplex
            fvals = [f0]
            for i in range(1, dim + 1):
                if call_count >= budget:
                    return simplex[0], fvals[0]
                if time.time() > deadline:
                    return simplex[0], fvals[0]
                fvals.append(f_count(simplex[i]))

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Centroid of all points except the worst
                worst_idx = dim
                centroid = np.mean(simplex[:dim], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[worst_idx])
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                fr = f_count(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[worst_idx])
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fe = f_count(xe)
                    if fe < fr:
                        simplex[worst_idx] = xe
                        fvals[worst_idx] = fe
                    else:
                        simplex[worst_idx] = xr
                        fvals[worst_idx] = fr
                elif fr < fvals[dim - 1]:
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
                else:
                    # Contraction
                    if fr < fvals[worst_idx]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[worst_idx])
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fc = f_count(xc)
                    if fc <= fr:
                        simplex[worst_idx] = xc
                        fvals[worst_idx] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if call_count >= budget:
                                return simplex[0], fvals[0]
                            if time.time() > deadline:
                                return simplex[0], fvals[0]
                            fvals[i] = f_count(simplex[i])

                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[dim]) < 1e-12:
                    break

            return simplex[0], fvals[0]

        # Run Nelder-Mead from the best starting point
        best_x, best_f = nelder_mead(best_x, best_f, max_iter=300)

        # If we have remaining budget, try a few more local refinements
        remaining = budget - call_count
        if remaining > 20 and time.time() < deadline:
            # Try Nelder-Mead from a few nearby points
            for attempt in range(min(5, remaining // 20)):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                # Perturb the current best
                perturbation = 0.5 * rng.standard_normal(dim)
                x_try = np.clip(best_x + perturbation, -5, 5)
                f_try = f_count(x_try)
                if f_try < best_f:
                    x_new, f_new = nelder_mead(x_try, f_try, max_iter=50)
                    if f_new < best_f:
                        best_x, best_f = x_new, f_new

        return best_x, best_f
# EVOLVE-BLOCK-END
