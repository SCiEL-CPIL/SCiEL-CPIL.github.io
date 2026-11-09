"""Optimized solver for BBOB f9 (rotated Rosenbrock) using CMA-ES + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES global search + L-BFGS-B high-precision polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 2.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        overall_best_f = np.inf
        overall_best_x = np.zeros(dim)

        # Phase 1: CMA-ES global search to locate the rotated Rosenbrock valley
        # CMA-ES adapts its covariance matrix to the anisotropic landscape,
        # efficiently navigating the narrow curved valley of Rosenbrock.
        if has_budget():
            try:
                # Start CMA-ES from origin with moderate sigma to cover [-5,5]^5
                # Population size ~ 4 + 3*ln(5) ≈ 10, but we use 8 for budget efficiency
                popsize = 8
                sigma0 = 2.0  # Initial step size to explore the domain
                max_evals_cma = min(300, budget - 80)  # Leave budget for polish

                # Manual CMA-ES implementation for precise budget control
                x_mean = np.zeros(dim)
                sigma = sigma0
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

                # Strategy parameters
                mu = popsize // 2  # Number of recombination weights
                # Rank-mu weights
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mueff = 1.0 / np.sum(w ** 2)
                c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2.0 / (dim + 1))
                c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
                mu_sigma = mueff / 2.0
                mu_c = mueff / 2.0
                chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

                # State variables
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                d = 1.0
                D = 1.0  # diagonal of C (initial)
                step_size = sigma0

                # Track best from CMA-ES
                cma_best_f = np.inf
                cma_best_x = np.zeros(dim)
                evals_so_far = 0

                while evals_so_far < max_evals_cma and has_budget():
                    if remaining_time() < 10.0:
                        break

                    # Generate population
                    population = np.zeros((popsize, dim))
                    for i in range(popsize):
                        z = rng.standard_normal(dim)
                        # Use C_sqrt * z for sampling
                        # For simplicity, use diagonal approximation initially
                        # then full C as it evolves
                        xi = x_mean + step_size * (C_sqrt @ z)
                        population[i] = xi

                    # Evaluate population
                    fitness = np.zeros(popsize)
                    for i in range(popsize):
                        if not has_budget():
                            fitness[i] = np.inf
                            continue
                        f = counted_func(population[i])
                        fitness[i] = f
                        evals_so_far += 1
                        if f < cma_best_f:
                            cma_best_f = f
                            cma_best_x = population[i].copy()

                    # Sort by fitness
                    order = np.argsort(fitness)
                    y = population[order]
                    fitness_sorted = fitness[order]

                    # Update mean
                    x_new = np.sum(w[:, None] * y[:mu], axis=0)

                    # Update step size control (sigma)
                    # Approximate: use diagonal of C
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ (x_new - x_mean) / sigma)
                    p_norm = np.linalg.norm(p_sigma)

                    if p_norm / (1 + 0.5 * (1 - d)) > chiN + 2.0 / dim:
                        step_size *= np.exp(0.8 / c_sigma)

                    # Update evolution path for covariance
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma

                    # Update covariance matrix C
                    # Rank-1 update
                    diff = (x_new - x_mean) / sigma
                    C = (1 + mu_sigma * (1 - c_c) * c_c + mu_sigma * c_c) * C \
                        + mu_sigma * np.outer(diff, diff)

                    # Rank-mu update
                    for i in range(mu):
                        yi = (y[i] - x_mean) / sigma
                        C += 2 * mu_sigma * w[i] * np.outer(yi, yi)

                    # Ensure C is positive definite and symmetric
                    C = (C + C.T) / 2.0
                    # Add small regularization
                    C += 1e-10 * np.eye(dim)

                    # Compute C_sqrt via Cholesky (with fallback)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                        C_sqrt_inv = np.linalg.inv(C_sqrt)
                    except np.linalg.LinAlgError:
                        # Fallback: use diagonal approximation
                        diag = np.sqrt(np.diag(C))
                        diag[diag < 1e-10] = 1e-10
                        C_sqrt = np.diag(1.0 / diag)
                        C_sqrt_inv = np.diag(diag)

                    x_mean = x_new
                    sigma = step_size

                    # Early termination if convergence
                    if np.all(np.abs(x_new - x_mean) < 1e-10):
                        break

                if cma_best_f < overall_best_f:
                    overall_best_f = cma_best_f
                    overall_best_x = cma_best_x.copy()

            except Exception:
                # Fallback: use tracked best
                if best_x[0] is not None:
                    overall_best_f = best_f[0]
                    overall_best_x = best_x[0].copy()

        # Update from global tracker
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0].copy()

        # Phase 2: L-BFGS-B high-precision polish from CMA-ES best
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Second L-BFGS-B polish with even tighter tolerance
        if has_budget() and remaining_time() > 3.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Final: use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
