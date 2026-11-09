"""Rewritten optimizer: CMA-ES with adaptive restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best across all restarts
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        lam = max(lam, 4)

        # Step size initialization
        sigma0 = 2.0

        # Max evaluations per restart
        max_evals_per_restart = 100

        calls_made = 0

        def cma_es(x0, sigma, budget_remaining):
            nonlocal calls_made, global_best_x, global_best_f

            n = dim
            # CMA-ES internal state
            mean = x0.copy()
            sigma = sigma
            C = np.eye(n)
            pc = np.zeros(n)
            cs = np.zeros(n)

            # Learning rates
            cmu = 1.0 / (4 * n + 4)
            cc = (4 - n + 4 / n) / ((n + 4) * (n + 4))
            c1 = 2 / ((n + 1.3) ** 2 + n)
            cmu = min(0.5, (2 + 2 / (n + 1)) / ((n + 2) ** 2 + n))
            # Actually standard: cmu = min(0.5, (2 + 2/(n+1)) / ((n+2)^2 + n))
            cmu = min(0.5, (2 + 2.0 / (n + 1)) / ((n + 2) ** 2 + n))

            chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            damps = 1 + 2 * max(0, np.sqrt((n + 1 - 2) / (n + 4) - 1)) + cc
            cc = (4 - n + 4.0 / n) / ((n + 4) * (n + 4))
            damps = 1 + 2 * max(0, np.sqrt((n + 1 - 2) / (n + 4) - 1)) + cc

            # Evaluate initial point
            fx0 = func(mean)
            calls_made += 1
            if fx0 < global_best_f:
                global_best_f = fx0
                global_best_x = mean.copy()

            fbest = fx0

            # Precompute C^{-1/2} for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            max_iter = budget_remaining
            for it in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                W = rng.standard_normal((lam, n))
                # Apply C^{-1/2} to each row
                Y = W @ C_inv_sqrt.T
                X = mean[None, :] + sigma * Y

                # Clip to box
                X = np.clip(X, -5, 5)

                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    if calls_made >= budget:
                        F[i:] = np.inf
                        break
                    if time.time() > deadline:
                        F[i:] = np.inf
                        break
                    F[i] = func(X[i])
                    calls_made += 1
                    if F[i] < global_best_f:
                        global_best_f = F[i]
                        global_best_x = X[i].copy()

                # Sort by fitness
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]

                # Update best
                if F_sorted[0] < fbest:
                    fbest = F_sorted[0]

                # Weighted mean
                weights = np.zeros(lam)
                mu = lam // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights[:mu] = w / w.sum()

                mean_new = (X_sorted[:mu] * weights[:, None]).sum(axis=0)

                # Update covariance
                delta_mean = mean_new - mean
                pc = (1 - cc) * pc + cc * (delta_mean / sigma)
                hsig = (np.sqrt(1 - (1 - cc) ** (2 * (it + 1))) / np.sqrt(1 - (1 - cc) ** 2)) * chi_n
                hsig = min(1.0, hsig)

                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc * (1 - cc) * C) + \
                    cmu * (X_sorted[:mu] - mean)[:, None].T @ (X_sorted[:mu] - mean) @ weights

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp((cc / damps) * (np.linalg.norm(pc) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Re-estimate C^{-1/2} every few iterations
                if it % 5 == 0 or it == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                mean = mean_new

                # Convergence check
                if np.linalg.norm(delta_mean) < 1e-12 * sigma:
                    break

            return mean, fbest

        # Multi-restart strategy
        # First, do a quick coarse scan to find promising regions
        # Then run CMA-ES from best starting points

        # Coarse grid: 3^5 = 243 points, too many. Use 2^5 = 32 corners + center
        # Actually let's use a smarter initialization
        # Start with a few diverse starting points

        n_starts = 5
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        # Sort starting points by function value (quick evaluation)
        start_fvals = []
        for sp in start_points:
            if calls_made >= budget:
                break
            fv = func(sp)
            calls_made += 1
            start_fvals.append(fv)
            if fv < global_best_f:
                global_best_f = fv
                global_best_x = sp.copy()

        # Sort by fitness
        order = np.argsort(start_fvals)
        start_points = [start_points[i] for i in order]
        start_fvals = [start_fvals[i] for i in order]

        # Run CMA-ES from best starting points
        for i in range(len(start_points)):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            remaining = budget - calls_made
            if remaining < 10:
                break

            # Use adaptive budget per restart
            # First restart gets more budget
            if i == 0:
                alloc = min(remaining, 200)
            elif i == 1:
                alloc = min(remaining, 100)
            else:
                alloc = min(remaining, 50)

            if alloc < 5:
                break

            sigma_init = 1.5 if i == 0 else 1.0
            cma_es(start_points[i], sigma_init, alloc)

            # If we've gotten very close, stop
            if global_best_f < 1e-6:
                break

        # If we have remaining budget, do a final local refinement
        if global_best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            if remaining >= 10 and time.time() < deadline:
                # Nelder-Mead style local search
                x = global_best_x.copy()
                fx = global_best_f
                # Simple coordinate descent / local search
                step = 0.5
                for _ in range(min(remaining, 50)):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            y = x.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls_made += 1
                            if fy < fx:
                                x, fx = y, fy
                                improved = True
                                if fx < global_best_f:
                                    global_best_f = fx
                                    global_best_x = x.copy()
                    if not improved:
                        step *= 0.5
                        if step < 1e-8:
                            break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
