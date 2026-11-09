"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical deterministic grid scan followed by localized CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to identify the global basin.
        # Use 3^5 = 243 points at {-5, 0, 5} for initial coverage.
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append([i * 5 - 5, j * 5 - 5, k * 5 - 5, 
                                                 l * 5 - 5, m * 5 - 5])

        # Evaluate all grid points
        best_x = None
        best_f = np.inf
        calls_used = 0
        for point in grid_points:
            x = np.array(point, dtype=float)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Localized CMA-ES around the best grid point
        remaining_budget = budget - calls_used
        if remaining_budget < 10:
            return best_x, best_f

        # Initialize CMA-ES parameters
        x = best_x.copy()
        sigma = 2.0  # Start with a moderate step size to explore locally
        C = np.eye(dim)
        L = np.eye(dim)  # Cholesky factor
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        cs = 1 / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + 10)
        mu = int(np.floor(4 * dim / 2))  # number of recombination points
        cmu = min(1 - c1, 2 * ((dim - 2) + 2 / (dim + 1)) / ((dim + 2) ** 2 + 15))
        weights = np.ones(mu)
        weights[0] = 2.0
        weights /= weights.sum()
        mu_eff = (weights.sum()) ** 2 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Deterministic RNG for reproducibility
        rng = np.random.default_rng(42)
        
        # CMA-ES loop
        while calls_used < budget:
            # Generate lambda_ offspring
            lambda_ = min(4 * dim, budget - calls_used)
            if lambda_ < 2:
                break
                
            offspring = []
            offspring_f = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                # Sample from N(x, sigma^2 * C) using Cholesky
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
                calls_used += 1
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = np.array([offspring[i] for i in indices])
            sorted_f = np.array([offspring_f[i] for i in indices])
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Update mean
            new_mean = np.sum(weights[:, None] * sorted_offspring[:mu], axis=0)
            
            # Update paths
            diff = (new_mean - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ diff)
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - cs) ** 2) < (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * diff
            
            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (norm_ps / chi_n - 1))
            
            # Update mean
            x = new_mean
            
            # Update Cholesky factor
            try:
                # Ensure positive definiteness
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal if not positive definite
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.1)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
