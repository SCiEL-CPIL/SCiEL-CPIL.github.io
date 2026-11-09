"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and restart rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        def cmaes(x0, sigma0, max_calls):
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_eig = np.eye(n)
            C_eigvals = np.ones(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            lam = 4
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            calls = 0
            best_x = x.copy()
            best_f = func(x)
            calls += 1

            gen = 0
            while calls < max_calls:
                # Generate population
                pops = np.zeros((lam, n))
                fits = np.full(lam, np.inf)
                for i in range(lam):
                    if calls >= max_calls:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fits[i] = func(y)
                    pops[i] = y
                    calls += 1

                if calls >= max_calls:
                    for i in range(lam):
                        if fits[i] < best_f:
                            best_f = fits[i]
                            best_x = pops[i].copy()
                    break

                # Sort by fitness
                order = np.argsort(fits)
                sorted_fits = fits[order]
                sorted_pops = pops[order]

                # Update best
                if sorted_fits[0] < best_f:
                    best_f = sorted_fits[0]
                    best_x = sorted_pops[0].copy()

                # Weighted mean of selected individuals
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (sorted_pops[i] - x)
                y_w = y_w / sigma

                # Update mean
                x = x + y_w * sigma

                # Update path ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w

                # Update path pc and covariance matrix C
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / c_sigma) * chi_n:
                    pc = (1 - c_c) * pc + c_c * y_w
                    delta = (np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma) * np.linalg.norm(ps)
                    # Update C
                    C = (1 + c_1 * delta + c_mu) * C
                    C -= c_1 * C
                    C += c_mu * (np.outer(pc, pc) - C)
                    # Add rank-mu update
                    for i in range(mu):
                        zi = (sorted_pops[i] - x) / sigma
                        C += c_mu * weights[i] * (np.outer(zi, zi))

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update eigen decomposition of C
                C_eigvals, C_eig = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-12)
                D = np.sqrt(C_eigvals)
                B = C_eig

                # Clip sigma
                sigma = np.clip(sigma, 1e-10, 1e10)

                # Restart condition: if sigma is too small or too large
                if sigma < 1e-8 or sigma > 1e8:
                    break

                gen += 1

            return best_x, best_f, calls

        # Run CMA-ES
        x0 = rng.uniform(-5, 5, n)
        best_x, best_f, used_calls = cmaes(x0, 2.0, budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
