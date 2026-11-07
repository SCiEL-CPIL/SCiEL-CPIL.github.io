"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: evaluator kills after 330s. Use 310s for safety.
        start_time = time.time()
        max_time = 310.0

        rng = np.random.default_rng(123)

        # Initialize CMA-ES
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of parents
        lambda_ = 2 * mu       # population size
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evaluate initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Track for restarts if needed, but let's keep it simple
        # We'll do a single CMA-ES run with the full budget

        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        count_steps = 0

        while calls < budget:
            # Check time
            if time.time() - start_time > max_time:
                break

            # Generate population
            W = B / D
            offspring = []
            offspring_f = []

            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (W @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                offspring_f.append(fy)
                calls += 1
                if time.time() - start_time > max_time:
                    break

            if len(offspring) == 0:
                break

            offspring_f = np.array(offspring_f[:len(offspring)])
            offspring = np.array(offspring)

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Only use top mu offspring for updates
            if len(offspring) < mu:
                break

            y_w = offspring[:mu]
            f_w = offspring_f[:mu]

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * y_w, axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * (W.T @ ((x - x_old) / sigma))))
            norm_ps = np.linalg.norm(ps)
            h_sig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - 1) - 1))) < (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu / mu_eff * np.sum(
                weights[:, None, None] * np.outer(y_w - x_old, y_w - x_old), axis=0) / sigma ** 2

            # Eigenvalue decomposition (periodically for stability)
            if calls % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(x - x_old) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
