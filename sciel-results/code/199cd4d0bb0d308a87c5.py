"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.full(dim, 0.0)
        best_f = float('inf')
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es_run(x0, sigma0, budget_run, rng):
            nonlocal call_count, best_x, best_f

            # CMA-ES parameters
            lambda_pop = 9
            mu = lambda_pop // 2

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Pre-compute B and D from C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            steps = 0
            while steps < budget_run // lambda_pop + 1:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                pop = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x = x_mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    f = safe_call(x)
                    pop.append((f, x))
                    if call_count >= budget:
                        break

                if call_count >= budget:
                    break

                # Sort population
                pop.sort(key=lambda t: t[0])

                # Update x_mean
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * pop[i][1]

                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new_mean - x_mean)

                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new_mean - x_mean)

                # Check if ||ps||/chi_n < (1.4 + 1/dim) * (1-c_sigma)
                if np.linalg.norm(ps) / chi_n < (1.4 + 1 / dim) * (1 - c_sigma):
                    C += c_1 * (np.outer(pc, pc) - C)

                # Update C
                C += c_mu * sum(weights[i] * np.outer(pop[i][1] - x_mean, pop[i][1] - x_mean) for i in range(mu))

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update x_mean
                x_mean = x_new_mean

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                B = eigvecs
                D = np.sqrt(eigvals)

                steps += 1

            return best_x, best_f

        # Restart strategy
        num_restarts = 5
        budget_per_restart = budget // num_restarts

        # First restart
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, dim)
        cma_es_run(x0, 1.0, budget_per_restart, rng)

        # Subsequent restarts
        for i in range(1, num_restarts):
            if call_count >= budget or time.time() > deadline:
                break
            # Start from best found so far
            x0 = best_x.copy()
            # Smaller sigma to exploit
            cma_es_run(x0, 0.5, budget_per_restart, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
