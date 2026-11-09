"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for rotated Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Initial parameters
        x = np.zeros(dim)  # start at origin
        sigma = 2.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        C_inv = np.eye(dim)
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        
        # CMA-ES adaptation rates
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Generate initial population
        def generate_population(x, sigma, C, rng):
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            z = rng.standard_normal((lam, dim))
            y = z @ L.T
            x_pop = x + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            return x_pop, y

        # Evaluate initial population
        x_pop, y_pop = generate_population(x, sigma, C, rng)
        f_pop = np.array([safe_func(x_pop[i]) for i in range(lam)])
        
        best_idx = np.argmin(f_pop)
        best_x = x_pop[best_idx].copy()
        best_f = f_pop[best_idx]

        # Main CMA-ES loop
        while call_count < budget and time.time() < deadline:
            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(f_pop)
            parents_idx = order[:mu]
            
            # Weighted recombination
            y_parents = y_pop[parents_idx]
            w_y = y_parents.T @ weights  # weighted sum of y
            x_new = x + sigma * w_y
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv @ (sigma * w_y)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count - lam))) < (1.4 + 2 / (dim + 1)) * chiN:
                pc = (1 - cs) * pc + cs * (C_inv @ x_new - x) / sigma
            
            # Update covariance matrix
            delta = (x_new - x) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 if np.linalg.norm(ps) ** 2 / (1 - (1 - cs) ** (2 * (call_count - lam))) > (1.4 + 2 / (dim + 1)) ** 2 * chiN ** 2 else 0) * C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', y_parents, y_parents, y_parents), axis=0)
            C = (C + C.T) / 2  # ensure symmetry
            
            # Update step size
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x = x_new
            
            # Generate new population
            x_pop, y_pop = generate_population(x, sigma, C, rng)
            f_pop = np.array([safe_func(x_pop[i]) for i in range(lam)])
            
            # Update best
            local_best_idx = np.argmin(f_pop)
            if f_pop[local_best_idx] < best_f:
                best_f = f_pop[local_best_idx]
                best_x = x_pop[local_best_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
