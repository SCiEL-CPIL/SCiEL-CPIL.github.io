"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Schaffers F7 in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        def track_call(x):
            nonlocal calls_used
            val = func(x)
            calls_used += 1
            return val

        # CMA-ES parameters
        lam = 12
        mu = lam // 2

        # Weights (logarithmic)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Variance-effectiveness
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = (2 + 2.0 / (dim + 1.0)) / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1 / mu_eff)) /
                 ((dim + 2) ** 2 + mu_eff))

        # Initial sigma
        sigma0 = 1.5
        x0 = rng.uniform(-4, 4, dim)

        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # We'll use a simple approach: sample via Cholesky of C
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def sample_from_C(n_samples):
            """Sample n_samples vectors from N(0, C) using Cholesky."""
            try:
                L = np.linalg.cholesky(C)
                Z = rng.standard_normal((n_samples, dim))
                return (Z @ L.T).T  # shape (n_samples, dim)
            except np.linalg.LinAlgError:
                # Fallback: use eigen-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                Z = rng.standard_normal((n_samples, dim))
                return (Z @ L.T).T

        def update_C_and_inverse():
            """Update C using rank-1 and rank-mu updates, then compute inverse Cholesky."""
            nonlocal C, C_sqrt_inv
            # Rank-1 update
            C += (1 + c_sigma * (1 - c_sigma)) * c1 * np.outer(p_c, p_c)
            # Rank-mu update
            y = (x_sorted[:mu] - x_mean) / sigma
            for i in range(mu):
                C += cmu * weights[i] * np.outer(y[i], y[i])
            # Ensure symmetry
            C = (C + C.T) / 2.0
            # Compute inverse Cholesky for sampling
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: use eigen-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                C_sqrt_inv = np.linalg.inv(L)

        # Main CMA-ES loop
        x_mean = x0.copy()
        fx_mean = track_call(x_mean)
        if fx_mean < best_f:
            best_f, best_x = fx_mean, x_mean.copy()

        generation = 0
        while calls_used < budget:
            if time.time() >= deadline:
                break

            # Sample population
            samples = sample_from_C(lam)
            population = x_mean + sigma * samples
            population = np.clip(population, -5, 5)

            # Evaluate
            fitness = np.empty(lam)
            for i in range(lam):
                if time.time() >= deadline:
                    break
                if calls_used >= budget:
                    break
                fitness[i] = track_call(population[i])
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = population[i].copy()

            if calls_used >= budget:
                break

            # Sort by fitness
            order = np.argsort(fitness)
            x_sorted = population[order]
            f_sorted = fitness[order]

            # Update mean
            x_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update p_sigma
            if np.linalg.norm(p_c) / np.sqrt(1 - (1 + generation / (c_sigma * (dim + 2))) ** 2) < (1.4 + 1 / dim) * chi_n:
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / sigma * (x_mean - x_sorted[0])
            else:
                p_sigma = (1 - c_sigma) * p_sigma

            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 + generation / (c_sigma * (dim + 2))) ** 2) <
                       (1.4 + 1 / dim) * chi_n) * 1.0
            c_c = (c1 + cmu) * (1 - h_sigma) + c_sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / sigma * (x_mean - x_sorted[0])

            # Update C
            update_C_and_inverse()

            # Update sigma (1/5 rule)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            generation += 1

        # Final local refinement if budget remains
        if best_x is not None and calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            recent = []
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                if calls_used >= budget:
                    break
                if len(recent) >= 10:
                    sr = sum(recent) / 10.0
                    if sr > 0.33:
                        sigma *= 1.1
                    elif sr < 0.1:
                        sigma *= 0.8
                    recent = recent[-10:]
                sigma = np.clip(sigma, 1e-6, 1.0)
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = track_call(y)
                if fy < fx:
                    x, fx = y, fy
                    recent.append(1)
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    recent.append(0)
                if len(recent) > 10:
                    recent = recent[-10:]

        return best_x, best_f
# EVOLVE-BLOCK-END
