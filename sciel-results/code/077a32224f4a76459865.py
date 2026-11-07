"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget or time.time() > deadline:
                return None
            calls_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, n_calls, rng):
            nonlocal best_f, best_x, calls_used

            n = dim
            # Population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Strategy parameters
            cs = (n + 1.3) / (n + 5)
            c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            d_amps = 1.0

            # Evaluate initial point
            fx = safe_call(x_mean)
            if fx is None:
                return x_mean, best_f

            # Main loop
            for gen in range(n_calls):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                off_f = []
                for y in offspring:
                    fy = safe_call(y)
                    if fy is None:
                        return x_mean, best_f
                    off_f.append(fy)

                # Sort by fitness
                indices = np.argsort(off_f)
                offspring_sorted = [offspring[i] for i in indices]
                off_f_sorted = [off_f[i] for i in indices]

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * offspring_sorted[i]
                x_new = y_w

                # Update sigma (CSA)
                if sigma > 1e-15:
                    z_sigma = (B.T @ (D * (B.T @ (x_new - x_mean)))) / sigma
                    p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_sigma
                    hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                    if hsig:
                        sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                    d_amps = (1 - cs) * d_amps + cs * (np.linalg.norm(p_sigma) / chi_n - 1)
                else:
                    hsig = False

                # Update C (covariance matrix)
                y_w_diff = x_new - x_mean
                x_mean = x_new
                p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w_diff

                # Update covariance matrix
                C_update = np.outer(p_c, p_c) - (1 - c1) * (D @ D)
                C_update += (1 - c1) * (1 + 2 * cmu / mu_eff - c1) * np.eye(n)
                for i in range(mu):
                    y_i = (B @ (D * (B.T @ (offspring_sorted[i] - x_mean)))) / sigma
                    C_update += w[i] * np.outer(y_i, y_i)

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C_update)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs

            return x_mean, best_f

        # Multi-restart strategy
        max_restarts = 8
        rng = np.random.default_rng(42)

        # First restart: start at origin with moderate sigma
        x0 = np.zeros(dim)
        sigma0 = 1.0
        calls_per_restart = max(30, budget // max_restarts)
        x_mean, best_f = cma_es(x0, sigma0, calls_per_restart, rng)

        # Subsequent restarts: start near best found, with varying sigmas
        for restart in range(1, max_restarts):
            if calls_used >= budget or time.time() > deadline:
                break

            # Initialize near best solution with some randomization
            if best_f < float('inf'):
                x0 = best_x + 0.5 * rng.standard_normal(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            x0 = np.clip(x0, -5, 5)

            # Vary sigma for diversity
            sigma0 = 0.5 * (1 + restart * 0.3)

            remaining = budget - calls_used
            if remaining <= 0:
                break

            x_mean, best_f = cma_es(x0, sigma0, min(calls_per_restart, remaining), rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
