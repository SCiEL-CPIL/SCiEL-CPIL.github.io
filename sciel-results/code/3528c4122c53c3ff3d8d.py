"""CMA-ES with restarts for Rastrigin rotated (f15)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Rastrigin rotated (f15)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cmaes_run(x0, sigma0, max_evals, rng):
            """Run one CMA-ES instance."""
            nonlocal calls, best_x, best_f
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)

            def update_C_sqrt(C):
                """Compute square root of covariance matrix using eigenvalue decomposition."""
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_sqrt, C_inv_sqrt

            C_sqrt, C_inv_sqrt = update_C_sqrt(C)

            local_best_x = x.copy()
            local_best_f = np.inf
            evals_used = 0
            stagnation = 0

            while evals_used < max_evals and calls < budget and time.time() < deadline:
                # Generate population
                population = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x_new = x + sigma * (C_sqrt @ z)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = evaluate(x_new)
                    evals_used += 1
                    population.append((f_new, x_new))

                if len(population) < lam:
                    break

                # Sort by fitness
                population.sort(key=lambda p: p[0])

                # Update best
                if population[0][0] < local_best_f:
                    local_best_f = population[0][0]
                    local_best_x = population[0][1].copy()

                # Select parents
                parents = [p[1] for p in population[:mu]]

                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * parents[i] for i in range(mu))

                # Update step size (CSA)
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ ((x - x_old) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (evals_used + 1) // lam))) / chi_n < 1.4 + 2.0 / (n + 1.0)
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update evolution paths and covariance
                if hsig:
                    pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (C_inv_sqrt @ ((x - x_old) / sigma))

                rank_mu = np.outer(parents, weights) @ parents.T
                rank_c = np.outer(pc, pc)
                C = (1.0 + c1 * (2.0 - c1) - cmu) * C + c1 * rank_c + cmu * rank_mu

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                C_sqrt, C_inv_sqrt = update_C_sqrt(C)

                # Stagnation detection
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 2 * lam:
                    break

            return local_best_x, local_best_f

        # Phase 1: Initial random sampling to find a good starting region
        n_init = 20
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        n_restarts = 10
        for restart in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Choose starting point
            if restart == 0:
                x0 = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)
                sigma0 = 1.5
            elif restart % 2 == 0:
                # Perturb best found
                x0 = best_x.copy() + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            else:
                # Random restart
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0

            # Estimate remaining budget
            remaining = budget - calls
            if remaining <= 0:
                break
            max_evals = min(remaining, 50)  # Cap per restart to allow multiple restarts

            local_x, local_f = cmaes_run(x0, sigma0, max_evals, rng)

            if local_f < best_f:
                best_f = local_f
                best_x = local_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
