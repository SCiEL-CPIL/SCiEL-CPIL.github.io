"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # 4 + 3*ln(5) ≈ 8
        # Learning rates
        mu = lam // 2  # number of best solutions used for mean update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial step size
        sigma = 2.0

        # Initial center: try a few starting points, pick best
        rng = np.random.default_rng(42)
        
        # Try a few random starts to find a good initial point
        best_x = None
        best_fx = np.inf
        for i in range(10):
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
        
        x_mean = best_x.copy()
        fx_best = best_fx
        best_x_global = best_x.copy()
        fx_global = best_fx

        # CMA-ES state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)  # Cholesky factor
        C_sqrt_inv = np.eye(dim)

        # Track best
        calls_made = 10  # we've already made 10 calls

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_fx = []
            for y in offspring:
                if time.time() > deadline:
                    break
                fy = func(y)
                offspring_fx.append(fy)
                calls_made += 1

            if len(offspring_fx) < lam:
                # Pad with NaN if we ran out of time mid-evaluation
                while len(offspring_fx) < lam:
                    offspring_fx.append(np.inf)

            # Sort by fitness
            indices = np.argsort(offspring_fx)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fx = [offspring_fx[i] for i in indices]

            # Update best
            if sorted_fx[0] < fx_best:
                fx_best = sorted_fx[0]
                best_x_global = sorted_offspring[0].copy()
                fx_global = fx_best

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_offspring[i]
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (new_mean - x_mean) / sigma)
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - 10 - lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - x_mean) / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            delta_vec = delta_hsig * pc * np.outer(pc, np.ones(dim))
            C = (1 + c_sigma * (c_sigma - c_sigma * d_sigma) + delta_hsig) * C
            for i in range(mu):
                yi = (sorted_offspring[i] - x_mean) / sigma
                C += c_sigma * weights[i] * np.outer(yi, yi)

            # Update step size (CSA)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) - 1 / (24 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Update C_sqrt via eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            C_sqrt_inv = np.linalg.inv(C_sqrt)

            x_mean = new_mean

            # Convergence check
            if sigma < 1e-10:
                break

        return best_x_global, fx_global
# EVOLVE-BLOCK-END
