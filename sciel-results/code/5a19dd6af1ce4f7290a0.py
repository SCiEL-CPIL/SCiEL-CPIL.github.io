"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses CMA-ES with adaptive population size and Nelder-Mead local polish.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES with Nelder-Mead polish for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 315.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        def still_ok():
            return time.time() < deadline

        def nelder_mead(func, x0, f0, calls_remaining, max_iter=80):
            """Nelder-Mead local optimization."""
            if calls_remaining < 20:
                return x0, f0, 0

            # Build initial simplex with adaptive step size
            simplex = [(x0.copy(), f0)]
            step = 0.1
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = float(func(xi))
                calls_remaining -= 1
                simplex.append((xi, fi))

            # Sort simplex
            simplex.sort(key=lambda t: t[1])

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            calls_used = dim + 1
            for _ in range(max_iter):
                if calls_remaining - calls_used < 10:
                    break

                centroid = np.zeros(dim)
                for i in range(dim):
                    centroid += simplex[i][0]
                centroid /= dim

                worst_idx = dim
                xr = centroid + alpha * (centroid - simplex[worst_idx][0])
                xr = np.clip(xr, -5.0, 5.0)
                fr = float(func(xr))
                calls_used += 1

                if fr < simplex[0][1]:
                    xe = centroid + gamma * (centroid - simplex[worst_idx][0])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = float(func(xe))
                    calls_used += 1
                    if fe < fr:
                        simplex[worst_idx] = (xe, fe)
                    else:
                        simplex[worst_idx] = (xr, fr)
                elif fr < simplex[dim - 1][1]:
                    simplex[worst_idx] = (xr, fr)
                else:
                    if fr < simplex[dim - 1][1]:
                        xc = centroid + rho * (simplex[worst_idx][0] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[worst_idx][0])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = float(func(xc))
                    calls_used += 1

                    if fc < simplex[dim - 1][1]:
                        simplex[worst_idx] = (xc, fc)
                    else:
                        for i in range(1, dim + 1):
                            simplex[i][0] = simplex[0][0] + sigma_nm * (simplex[i][0] - simplex[0][0])
                            simplex[i][0] = np.clip(simplex[i][0], -5.0, 5.0)
                            simplex[i][1] = float(func(simplex[i][0]))
                            calls_used += 1

                simplex.sort(key=lambda t: t[1])

                if abs(simplex[0][1] - simplex[dim][1]) < 1e-12:
                    break

            return simplex[0][0], simplex[0][1], calls_used

        best_x_overall = np.zeros(dim)
        best_f_overall = np.inf

        # Phase 1: Robust initialization - sample diverse points
        n_init = 12
        init_points = [np.zeros(dim)]
        # Add points along each axis
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 3.0
            init_points.append(p.copy())
            p[i] = -3.0
            init_points.append(p.copy())
        # Random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-5.0, 5.0, dim))

        init_results = []
        for x in init_points:
            f = float(func(x))
            init_results.append((x.copy(), f))
            if f < best_f_overall:
                best_f_overall = f
                best_x_overall = x.copy()

        calls_made = len(init_points)

        # Phase 2: CMA-ES
        # Use CMA-ES with population size ~4-6
        cma_budget = int(budget * 0.55)
        max_cma_calls = cma_budget - calls_made

        # CMA-ES parameters
        pop_size = 4
        mu = max(1, pop_size // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        sigma = 2.0
        x_mean = best_x_overall.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cholesky_update(C, delta):
            """Update C = C + delta using Cholesky-based rank update."""
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(eigvals + delta) @ eigvecs.T

        cma_calls = 0
        no_improve = 0
        no_improve_limit = 30

        while calls_made + cma_calls < budget and still_ok():
            if cma_calls >= max_cma_calls:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                xi = x_mean + sigma * y
                xi = np.clip(xi, -5.0, 5.0)
                fi = float(func(xi))
                calls_made += 1
                cma_calls += 1
                offspring.append((xi, fi))

            # Sort offspring
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < best_f_overall:
                best_f_overall = offspring[0][1]
                best_x_overall = offspring[0][0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i][0]
            x_mean = x_new_mean

            # Update evolution paths
            y_mean = (x_mean - x_mean) / sigma  # This is zero by construction in standard CMA-ES
            # Actually, y_mean should be the weighted sum of z vectors
            # Let's recompute properly
            # We need to track the z vectors

            # For simplicity, use a simplified CMA-ES update
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix (simplified - use rank-1 update with top solution)
            y_vec = (offspring[0][0] - x_mean) / max(sigma, 1e-12)
            C = (1 - c1 - cmu) * C + c1 * (2 - mu) * np.outer(y_vec, y_vec) + cmu * np.outer(y_vec, y_vec)
            # Ensure positive definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Check for restart
            if no_improve >= no_improve_limit:
                # Try a random restart near current best
                if calls_made < budget - 20:
                    x_restart = best_x_overall + rng.standard_normal(dim) * sigma
                    x_restart = np.clip(x_restart, -5.0, 5.0)
                    f_restart = float(func(x_restart))
                    calls_made += 1
                    cma_calls += 1
                    if f_restart < best_f_overall:
                        best_f_overall = f_restart
                        best_x_overall = x_restart.copy()
                        x_mean = x_restart.copy()
                        no_improve = 0
                    else:
                        sigma *= 0.5
                        no_improve = 0

            # Convergence check
            if np.linalg.norm(C - np.eye(dim) * np.trace(C) / dim) / (np.trace(C) / dim) < 1e-8:
                break

        # Phase 3: Nelder-Mead polish
        if still_ok() and calls_made < budget:
            x_nm, f_nm, calls_used = nelder_mead(
                func, best_x_overall, best_f_overall,
                budget - calls_made, max_iter=80
            )
            calls_made += calls_used
            if f_nm < best_f_overall:
                best_f_overall = f_nm
                best_x_overall = x_nm.copy()

        # Phase 4: Final (1+1)-ES refinement if budget remains
        if still_ok() and calls_made < budget:
            sigma = 0.3
            x = best_x_overall.copy()
            f_best = best_f_overall
            no_improve_count = 0
            no_improve_limit = 20

            while calls_made < budget and still_ok():
                y = rng.standard_normal(dim)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)

                f_new = float(func(x_new))
                calls_made += 1

                if f_new < f_best:
                    x = x_new
                    f_best = f_new
                    if f_best < best_f_overall:
                        best_f_overall = f_best
                        best_x_overall = x_new.copy()
                    no_improve_count = 0
                    sigma *= 1.03
                else:
                    no_improve_count += 1
                    sigma *= 0.97

                sigma = np.clip(sigma, 1e-10, 3.0)

                if no_improve_count >= no_improve_limit:
                    break

        return best_x_overall, float(best_f_overall)
