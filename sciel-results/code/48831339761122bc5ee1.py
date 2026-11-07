"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2  # number of parents
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial population: spread out
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Evaluate initial point
        f_count = 0
        fx = func(x_mean)
        f_count += 1
        best_x = x_mean.copy()
        best_f = fx

        # Generate initial population
        B = np.linalg.cholesky(C)
        D = np.diag(np.sqrt(np.diag(C)))
        D_inv = 1.0 / np.sqrt(np.diag(C))

        # Precompute for efficiency
        for gen in range(budget // lam):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate population
            fvals = []
            for y in population:
                if time.time() > deadline - 2.0:
                    # Fill remaining with worst possible
                    fvals.append(np.inf)
                    continue
                fy = func(y)
                f_count += 1
                fvals.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            fvals = np.array(fvals)

            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = [population[i] for i in idx]
            f_sorted = fvals[idx]

            # Update mean
            y_weighted = np.zeros(dim)
            for i in range(mu):
                y_weighted += w[i] * y_sorted[i]
            x_new = x_mean + c_c * (y_weighted - x_mean)

            # Update evolution paths
            d = np.sqrt(c_1 * (2 - c_1) * mu_eff * (1 - c_c))
            pc = (1 - c_c) * pc + c_c * (y_weighted - x_mean) / (sigma * d)
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (f_count / lam + 1)) / chi_n) < (1 + 2 / (dim + 1)) * 1.4

            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (y_weighted - x_mean)

            # Update covariance matrix
            delta_hsig = (1.0 - hsig) * c_sigma * (2 - c_sigma)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            for i in range(mu):
                zi = (y_sorted[i] - x_mean) / sigma
                C += c_1 * np.outer(zi, zi)
                C += c_mu * w[i] * np.outer(zi, zi)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = np.clip(x_new, -5, 5)

            # If we've used all budget, break
            if f_count >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
