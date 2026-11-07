"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            calls_used += 1
            return func(x)

        def cma_es(x0, sigma0, budget_cma, rng):
            """Run CMA-ES from x0 with initial sigma0, using at most budget_cma calls."""
            nonlocal best_x, best_f, calls_used

            n = dim
            # Population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Parameters
            c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            cs = (n + 1.3) / (n + 5)
            c_sigma = cs
            c_c = (4 + mu_eff / c1) / (n + 4)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            def update_distribution(mean, sigma, B, D, C, p_sigma, p_c):
                """Update B, D from C via eigen decomposition."""
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                return B, D

            # Initial evaluation
            fx = safe_call(mean)
            if fx is None:
                return
            if fx < best_f:
                best_f, best_x = fx, mean.copy()

            remaining = budget_cma - calls_used
            if remaining <= 0:
                return

            # CMA-ES main loop
            while calls_used < budget_cma and time.time() < deadline:
                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                off_f = []
                for y in offspring:
                    fy = safe_call(y)
                    if fy is None:
                        return
                    off_f.append(fy)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

                # Sort by fitness
                indices = np.argsort(off_f)

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * offspring[indices[i]]
                new_mean = y_w

                # Compute weighted sum in transformed space
                # z_w = B.T @ (D * (B.T @ (new_mean - mean))) / sigma
                diff = new_mean - mean
                z_w = (B.T @ (D * (B.T @ diff))) / sigma if sigma > 1e-15 else np.zeros(n)

                # Update sigma (CSA)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                if hsig:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C (covariance matrix)
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_w

                # Covariance update
                C_update = (1 + c1 * (cmu / mu_eff - 1) - c_c * (1 - c_c)) * C
                C_update += c_c * (1 - c_c) * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    y_i = (B.T @ (D * (B.T @ (offspring[indices[i]] - new_mean)))) / sigma
                    C_update += w[i] * np.outer(y_i, y_i)

                C = C_update

                # Update mean
                mean = new_mean

                # Update B, D
                B, D = update_distribution(mean, sigma, B, D, C, p_sigma, p_c)

                # Convergence check
                if np.linalg.norm(D) < 1e-10 or sigma < 1e-10:
                    break

            return

        # Multi-restart strategy
        rng = np.random.default_rng(42)

        # Phase 1: CMA-ES from origin with moderate sigma
        cma_es(np.zeros(dim), 1.5, 200, rng)

        # Phase 2: CMA-ES from random points
        for i in range(3):
            if calls_used >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-3, 3, dim)
            cma_es(x0, 1.0, 100, rng)

        # Phase 3: Local refinement around best found
        if calls_used < budget and time.time() < deadline:
            cma_es(best_x, 0.1, budget - calls_used, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
