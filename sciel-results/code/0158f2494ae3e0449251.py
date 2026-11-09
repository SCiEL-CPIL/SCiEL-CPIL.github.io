"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for Bent Cigar with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def check_deadline():
            return time.time() > deadline

        # CMA-ES parameters
        # For 5D, population size lambda is typically 4 + floor(3 * ln(dim)) = 4 + floor(3*1.609) = 4+4 = 8
        # But we have a budget of 500 calls, so we can use a larger population
        # Let's use lambda = 16 for better exploration
        lam = 16
        mu = lam // 2  # number of parents
        # Weights: linear weights, normalized
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        c_mu = (mu_eff - 2) / ((dim + 2) + mu_eff - 2)
        mu_eff = 1 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_ = min(c_1, 2 * (1 - c_1) * c_mu / (dim + 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial center: try a few random points to find a good starting point
        rng = np.random.default_rng(42)
        
        # Phase 0: Quick random search to find a good initial center
        best_x = None
        best_f = np.inf
        calls = 0

        # Try the origin and a few random points
        test_points = [np.zeros(dim)]
        for _ in range(10):
            test_points.append(rng.uniform(-5, 5, dim))
        
        for x in test_points:
            if check_deadline() or calls >= budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # CMA-ES state
        x_mean = best_x.copy() if best_x is not None else np.zeros(dim)
        sigma = 1.0  # initial step size
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Precompute C^{-1/2} initially
        eigvals, eigvecs = np.linalg.eigh(C)
        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

        # Main CMA-ES loop
        while not check_deadline() and calls < budget:
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if check_deadline() or calls >= budget:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                offspring.append(x)
                offspring_f.append(fx)

            if len(offspring) == 0:
                break

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = [offspring[i] for i in order]
            offspring_f = [offspring_f[i] for i in order]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Select parents
            y_parents = []
            for i in range(mu):
                y = (offspring[i] - x_mean) / sigma
                y_parents.append(y)
            y_parents = np.array(y_parents)

            # Update mean
            x_mean_new = np.sum(weights[:, None] * y_parents, axis=0) * sigma + x_mean
            x_mean = np.clip(x_mean_new, -5, 5)

            # Update step size (sigma)
            y_sigma = (C_inv_sqrt @ (np.sum(weights[:, None] * y_parents, axis=0) * sigma)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            y_parents_centered = y_parents - np.mean(y_parents, axis=0)
            delta_sig = np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)) - 1
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_new) / sigma
            C = (1 + c_1 * (mu_eff - 2) + c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y_parents, y_parents), axis=0)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

            # Check convergence
            if sigma < 1e-10:
                break

            # If we've been stuck, try a restart with larger sigma
            if np.std(offspring_f[:mu]) < 1e-8:
                sigma *= 2
                if sigma > 10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
