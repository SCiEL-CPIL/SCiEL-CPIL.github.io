"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            calls_used += 1
            val = func(x)
            return val

        def cma_es(x0, sigma0, max_evals, rng):
            """Run CMA-ES starting from x0 with sigma0, using at most max_evals evaluations."""
            nonlocal calls_used, best_f, best_x

            n = dim
            # Population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Strategy parameters
            c_sigma = (n + 1.3) / (n + 5)
            c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            evals_in_this_run = 0

            while evals_in_this_run < max_evals:
                if time.time() > deadline:
                    break
                remaining_budget = budget - calls_used
                if remaining_budget <= 0:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    offspring.append(y)

                # Evaluate offspring
                off_f = []
                for y in offspring:
                    if calls_used >= budget:
                        break
                    fy = safe_call(y)
                    if fy is None:
                        off_f = None
                        break
                    off_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if off_f is None or len(off_f) < lam:
                    break

                evals_in_this_run += len(off_f)

                # Sort by fitness
                indices = np.argsort(off_f)
                sorted_offspring = [offspring[i] for i in indices]
                sorted_f = [off_f[i] for i in indices]

                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += w[i] * sorted_offspring[i]

                # Compute z_sigma for CSA
                # z_sigma = B^T * D * B^T * (new_mean - mean) / sigma
                diff = new_mean - mean
                z_sigma = (B.T @ (D * (B.T @ diff))) / sigma

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_in_this_run + 1)))) < (1.4 + 2 / (n + 1)) * chi_n

                # Update sigma
                if h_sigma:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Update p_c and C
                p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mu_eff) * diff

                # C update
                C_new = (1 - c1 * (2 - c1)) * C
                C_new += c1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    y_i = (B.T @ (D * (B.T @ (sorted_offspring[i] - new_mean)))) / sigma
                    C_new += w[i] * np.outer(y_i, y_i)

                # Eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C_new)
                eigvals = np.clip(eigvals, 1e-15, None)
                B = eigvecs
                D = np.sqrt(eigvals)

                mean = new_mean

            return mean, sigma

        # Restart strategy
        rng = np.random.default_rng(42)

        # Phase 1: Explore from origin with moderate sigma
        remaining = budget
        alloc1 = min(remaining, max(50, budget // 3))
        mean1, sigma1 = cma_es(np.zeros(dim), 2.0, alloc1, rng)
        remaining -= alloc1

        # Phase 2: Explore from a few random points
        if remaining > 20:
            alloc2 = min(remaining, max(30, budget // 4))
            for _ in range(2):
                if remaining <= 0 or time.time() > deadline:
                    break
                x_start = rng.uniform(-4, 4, dim)
                alloc_per = alloc2 // 2
                _, _ = cma_es(x_start, 1.5, alloc_per, rng)
                remaining -= alloc_per

        # Phase 3: Refine around best found
        if remaining > 10 and time.time() < deadline:
            alloc3 = remaining
            _, _ = cma_es(best_x.copy(), 0.5, alloc3, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
