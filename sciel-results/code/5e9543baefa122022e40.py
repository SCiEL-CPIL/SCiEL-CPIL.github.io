"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES and CMA-ES optimizer for the bent cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: (1+1)-ES for fast initial exploration
        # Use 10% of budget for this phase
        phase1_budget = max(10, budget // 10)
        
        x = rng.uniform(-5, 5, dim)
        fx = evaluate(x)
        sigma = 2.0

        for i in range(phase1_budget - 1):
            if time.time() > deadline or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        # Phase 2: CMA-ES for refined anisotropic search
        # Initialize CMA-ES with the best point found
        x_mean = x.copy()
        sigma = max(sigma * 0.5, 0.1) # Start CMA-ES with a smaller sigma
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))

            population = []
            fvals = []
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_child = x_mean + sigma * L @ z
                x_child = np.clip(x_child, -5, 5)
                f_child = evaluate(x_child)
                population.append(x_child)
                fvals.append(f_child)

            if len(population) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            population = [population[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i]

            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * L @ (x_new - x_mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution paths
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

            # Update covariance matrix
            delta = hsig * (1 - c_c) + c_c
            outer = np.outer(pc, pc)
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * (outer - C)
            for i in range(mu):
                diff = (population[i] - x_mean) / sigma
                C += c_mu * weights[i] * (np.outer(diff, diff) - C)

            # Symmetrize C
            C = (C + C.T) / 2

            # Update mean
            x_mean = x_new

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
