"""Rewritten optimizer for BBOB-style f12 (Bent cigar) with deadline guard."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        def remaining():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def cma_es(x0, f0, sigma0, budget_cma, rng, func, deadline):
            """Run CMA-ES from x0 with given budget, respecting deadline."""
            dim = self.dim
            # CMA-ES parameters
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1)**2 + 2)
            cmu = min(1 - c1, (2 + 2.0 / (dim + 1.3) - (dim + 2) / (dim + 4)) / ((dim + 1)**2 + 2))
            cs = (dim + 1.7) / (dim + 9 + c1)
            ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
            cc = (4 + dim / dim) / (4 + dim)  # path length adaptation for mean

            x_mean = x0.copy()
            f_mean = f0
            C = np.eye(dim)
            sigma = sigma0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # Covariance matrix update helper
            def inv_sqrt(C):
                # Use eigenvalue decomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return inv_sqrt

            # Initial step
            x = x_mean.copy()
            f = f_mean
            best_local_x = x.copy()
            best_local_f = f

            for gen in range(budget_cma):
                if time.time() > deadline - 5.0:
                    break

                # Generate offspring
                # Use Cholesky-like factorization via eigenvalues
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
                D = np.diag(np.sqrt(eigvals))

                ys = rng.standard_normal((lam, dim))
                # Apply B to each y
                offspring = np.array([x_mean + sigma * (B @ y) for y in ys])
                # Clip to box
                offspring = np.clip(offspring, -5, 5)

                # Evaluate
                fitnesses = np.array([func(x_off) for x_off in offspring])

                # Sort by fitness
                order = np.argsort(fitnesses)
                mu = lam // 2  # number of recombination points
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)

                # Update mean
                x_new_mean = np.sum(weights[:, None] * offspring[order[:mu]], axis=0)

                # Update path for sigma
                y_sigma = inv_sqrt(C) @ (x_new_mean - x_mean) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update path for covariance
                y_c = inv_sqrt(C) @ (x_new_mean - x_mean)
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_c

                # Rank-mu update
                delta_mu = (np.linalg.norm(p_c) / chi_n)**2 - 1
                C = (1 + c1 * delta_mu + cmu) * C
                C -= (c1 + cmu) * np.outer(p_c, p_c)
                C += cmu * sigma**2 * np.sum(
                    [weights[i] * np.outer(offspring[order[i]] - x_mean, offspring[order[i]] - x_mean) for i in range(mu)],
                    axis=0
                ) / sigma**2 if sigma > 0 else 0

                # Symmetrize
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new_mean

                # Track best
                for i in range(lam):
                    if fitnesses[i] < best_local_f:
                        best_local_f = fitnesses[i]
                        best_local_x = offspring[i].copy()

                # Early stopping check
                if gen > 10:
                    # Check if sigma is too small or no progress
                    if sigma < 1e-10:
                        break

            return best_local_x, best_local_f

        # Multi-start strategy
        # Phase 1: Several CMA-ES runs from different starting points
        n_starts = 5
        budget_per_start = max(10, budget // n_starts - 10)

        # Initial random points
        start_points = []
        # Include origin, random points, and points near boundaries
        start_points.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        # Shuffle starts
        rng.shuffle(start_points)

        for i, x0 in enumerate(start_points):
            if time.time() > deadline - 30.0:
                break

            f0 = func(x0)
            update_best(x0, f0)

            # Adaptive sigma based on initial function value
            sigma0 = 1.5 if abs(f0) < 10 else 2.0

            # Run CMA-ES
            x_res, f_res = cma_es(x0, f0, sigma0, budget_per_start, rng, func, deadline)
            update_best(x_res, f_res)

        # Phase 2: Local polish using Nelder-Mead on the best point
        if time.time() < deadline - 15.0 and best_x is not None:
            try:
                from scipy.optimize import minimize

                def wrapper(x):
                    return func(np.clip(x, -5, 5))

                # Nelder-Mead polish
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_f:
                    update_best(np.clip(res.x, -5, 5), res.fun)
            except Exception:
                pass

        # Phase 3: Final fine-grained search around best point
        if time.time() < deadline - 10.0 and best_x is not None:
            # Small random perturbations
            sigma_fine = 0.1
            for _ in range(min(50, int(remaining() / 0.01))):
                if time.time() > deadline - 5.0:
                    break
                x_try = np.clip(best_x + sigma_fine * rng.standard_normal(dim), -5, 5)
                f_try = func(x_try)
                if f_try < best_f:
                    update_best(x_try, f_try)
                    # Shrink sigma
                    sigma_fine *= 0.9
                else:
                    sigma_fine *= 0.8

        return best_x, best_f
