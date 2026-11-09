"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f18.
    Population-based with second-order adaptation of the search distribution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin before 330s limit

        # CMA-ES parameters
        pop_size = 4 * dim + 2  # standard recommendation
        lam = pop_size
        mu = lam // 2
        # Weights: linear, normalized
        w_raw = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Track best
        best_x = x.copy()
        best_fx = func(x)
        calls_used = 1

        # Precompute B and D via eigendecomposition helper
        def update_BD():
            nonlocal B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

        update_BD()

        while calls_used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            fitness = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline - 2.0:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring[i] = y
                fitness[i] = func(y)
                calls_used += 1
                if fitness[i] < best_fx:
                    best_fx = fitness[i]
                    best_x = y.copy()

            # Sort by fitness
            idx = np.argsort(fitness)
            sorted_offspring = offspring[idx]
            sorted_fitness = fitness[idx]

            # Weighted recombination
            y_w = np.sum(w[:, None] * sorted_offspring[:mu], axis=0)
            x_old = x.copy()
            x = x + y_w

            # Step-size control (CSA)
            y_sigma = (B @ (D * ((B.T @ (x - x_old)) / sigma))) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_used / (lam * d_sigma)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Covariance matrix adaptation
            yc = (x - x_old) / sigma
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * yc
            C = (1 + c_1 * (hsig - 1) * c_c * (2 - c_c) - c_1 - c_mu) * C
            C += c_1 * np.outer(pc, pc)
            C += c_mu * np.sum(w[:, None, None] * np.outer(sorted_offspring[:mu] - x_old, sorted_offspring[:mu] - x_old) / sigma ** 2, axis=0)

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Project to PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            update_BD()

            # Safety: if sigma explodes or becomes too small, reset
            if sigma > 100 or sigma < 1e-8:
                sigma = 2.0
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                x = best_x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
