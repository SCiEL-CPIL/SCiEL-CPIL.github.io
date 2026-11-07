"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(1.0 / np.log(2 * dim + 2)))  # population size
        c_sigma = (mu_eff + 1) / ((dim + 1) ** 2 + 2 * mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mu = lambda_ // 2  # number of parents
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w[-1] = 0
        mu_eff = 1 / np.sum(w ** 2)

        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        # Covariance matrix update state
        C = np.eye(dim)

        # Precompute
        alpha_sigma = (mu_eff + 2) / (dim + mu_eff)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        dampc = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_c

        for iteration in range(budget):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = x + sigma * (B @ (D * z))

            # Evaluate population
            fits = np.zeros(lambda_)
            for i in range(lambda_):
                fits[i] = func(pop[i])

            # Sort by fitness
            idx = np.argsort(fits)
            pop_sorted = pop[idx]
            fits_sorted = fits[idx]

            # Update best
            if fits_sorted[0] < best_fx:
                best_fx = fits_sorted[0]
                best_x = pop_sorted[0].copy()

            # Compute new mean
            x_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)

            # Update path evolution strategy
            y = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y))
            norm_ps = np.linalg.norm(ps)

            # Update covariance matrix
            delta_hs = (norm_ps / chi_n > 1 + 4 / (dim + 1))
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) - C)
            C += c_mu * np.sum(w[:, None, None] * np.outer(pop_sorted[:mu] - x, pop_sorted[:mu] - x) / sigma ** 2, axis=0)

            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            B = B.T  # Columns are eigenvectors

            # Update path coordinate system
            pc = (1 - c_c) * pc + c_c * y

            # Update sigma
            sigma *= np.exp(c_sigma / damps * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update mean
            x = x_new

            # Track best
            if fits_sorted[0] < best_fx:
                best_fx = fits_sorted[0]
                best_x = pop_sorted[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
