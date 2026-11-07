"""Rewritten optimizer: multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for black-box continuous optimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def try_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start CMA-ES
        n_restarts = 3
        budget_per_restart = max(1, budget // n_restarts)

        for restart in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Start near best found, with some perturbation
                x0 = best_x + rng.standard_normal(dim) * 1.0
                x0 = np.clip(x0, -5, 5)

            x = x0.copy()
            fx = try_func(x)
            sigma = 2.0
            pop_size = max(4, dim + 2)  # lambda
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            mu = pop_size // 2
            weights = np.log(pop_size + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for gen in range(budget_per_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                ys = []
                fvals = []
                for i in range(pop_size):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = x + sigma * C @ rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = try_func(y)
                    ys.append(y)
                    fvals.append(fy)

                if len(ys) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                ys = np.array(ys)
                fvals = np.array(fvals)
                ys = ys[order]
                fvals = fvals[order]

                # Update mean
                w = weights[:len(ys)]
                x_new = np.sum(w[:, None] * ys[:len(ys)], axis=0)

                # Update sigma (CSA)
                d_norm = np.linalg.norm(x_new - x)
                if d_norm > 1e-12:
                    cs = (1 - c_sigma) * cs + c_sigma * (x_new - x) / (sigma * chi_n)
                    h_sigma = (np.linalg.norm(cs) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(cs) / chi_n - 1))
                else:
                    h_sigma = True

                # Update C
                pc = (1 - c_c) * pc + c_c * h_sigma * (x_new - x) / sigma
                delta_pc = (1 - c_c) ** 0.5 if h_sigma else 0
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * np.sum(w[:, None, None] * (ys[:len(ys)] - x)[:, None, :] * (ys[:len(ys)] - x)[:, :, None] / sigma ** 2, axis=0)

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Ensure sigma is reasonable
                sigma = np.clip(sigma, 1e-4, 10.0)

                x = x_new

            # After each restart, refine around best found
            if calls_made < budget and time.time() < deadline:
                # Local refinement with smaller steps
                x_local = best_x.copy()
                sigma_local = 0.5
                for _ in range(min(50, budget - calls_made)):
                    if time.time() > deadline:
                        break
                    y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = try_func(y)
                    if fy < best_f:
                        x_local = y
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.8
                    if sigma_local < 0.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
