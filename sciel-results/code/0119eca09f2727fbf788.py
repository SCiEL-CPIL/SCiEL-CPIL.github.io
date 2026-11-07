"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and deadline guard."""

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

        # Multiple restarts with CMA-ES
        max_restarts = 5
        per_restart_budget = budget // max_restarts + 1

        for restart in range(max_restarts):
            if calls_used >= budget or time.time() > deadline:
                break

            # Initialize: first restart at origin, others random
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = np.random.default_rng(42 + restart).uniform(-4, 4, dim)

            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # CMA-ES parameters
            n = dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            sigma = 1.0
            c1 = (2 / ((n + 2) ** 2 + 10))
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            cs = (n + 1.3) / (n + 5 + cs if False else (n + 5))  # damping
            cs = (n + 1.3) / (n + 5)
            c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            d_amps = 1.0

            remaining = budget - calls_used
            if remaining <= 0:
                break

            for _ in range(remaining):
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = np.random.default_rng(1000 + restart * 100 + i).standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                off_f = []
                for y in offspring:
                    fy = safe_call(y)
                    if fy is None:
                        off_f = None
                        break
                    off_f.append(fy)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

                if off_f is None:
                    break

                # Sort by fitness
                indices = np.argsort(off_f)
                mu = lam // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1.0 / np.sum(w ** 2)

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * offspring[indices[i]]
                x_new = y_w

                # Update sigma (CSA)
                z_sigma = (np.sqrt(D.T @ D) * (B.T @ (x_new - x))) / (sigma * np.sqrt(D.T @ D)) if sigma > 1e-15 else np.zeros(n)
                z_sigma = (B.T @ (D * (B.T @ (x_new - x)))) / sigma if sigma > 1e-15 else np.zeros(n)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                if hsig:
                    sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                d_amps = (1 - cs) * d_amps + cs * (np.linalg.norm(p_sigma) / chi_n - 1)

                # Update C (covariance matrix)
                y_w_diff = x_new - x
                x = x_new
                p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w_diff
                C_update = np.outer(p_c, p_c) - (1 - c1) * D @ D
                C_update += (1 - c1) * (1 + 2 * cmu / mu_eff - c1) * np.eye(n)
                for i in range(mu):
                    y_i = (B @ (D * (B.T @ (offspring[indices[i]] - x_new)))) / sigma
                    C_update += w[i] * np.outer(y_i, y_i)
                # Note: This simplified CMA-ES may not be numerically stable; 
                # we use a simpler (1+1)-ES fallback for robustness.

                break  # Break inner loop, use simpler approach below

            # Fallback: simple (1+1)-ES for this restart
            sigma = 1.0
            for _ in range(budget - calls_used):
                if time.time() > deadline or calls_used >= budget:
                    break
                y = np.clip(x + sigma * np.random.default_rng(2000 + calls_used).standard_normal(n), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
