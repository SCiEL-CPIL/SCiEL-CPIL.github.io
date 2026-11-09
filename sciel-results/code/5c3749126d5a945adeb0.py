"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid grid-search warm-up + CMA-ES refinement for Sphere minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Phase 1: Deterministic grid-search warm-up to bracket the minimum
        # Use a coarse grid with ~100 evaluations to locate the basin
        grid_points = []
        # Create a 3-level grid in each dimension (27 points for dim=5)
        # Plus some additional points for better coverage
        levels = [-4.0, -2.0, 0.0, 2.0, 4.0]
        
        # Evaluate all grid points
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Generate grid points (5^5 = 3125 is too many, use subset)
        # Use a smarter subset: evaluate center, then expand
        # Start with center
        x_center = np.zeros(dim)
        fx_center = func(x_center)
        calls_made += 1
        best_x = x_center.copy()
        best_f = fx_center
        
        # Evaluate axis-aligned points at various distances
        axis_points = []
        for i in range(dim):
            for d in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = d
                axis_points.append(p)
        
        # Evaluate these axis points
        for p in axis_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # Evaluate some diagonal points
        diag_points = []
        for sign in [-1, 1]:
            for d in [2.0, 4.0]:
                p = np.full(dim, sign * d)
                diag_points.append(p)
        
        for p in diag_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # Phase 2: CMA-ES refinement starting from the best grid point
        # Use CMA-ES with the bounding box from grid search as initial distribution
        
        # CMA-ES parameters
        x = best_x.copy()
        f_x = best_f
        sigma = 1.5  # Initial step size
        
        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        mu = dim // 2 + 1  # Number of parents
        lambda_ = 2 * dim  # Population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # CMA-ES main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate lambda candidates
            B = np.linalg.cholesky(C)
            candidates = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                candidates.append(y)
            
            # Evaluate candidates
            f_vals = []
            for y in candidates:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                fy = func(y)
                f_vals.append(fy)
                calls_made += 1
            
            if len(f_vals) < lambda_:
                break
            
            # Sort by fitness
            idx = np.argsort(f_vals)
            y_sorted = [candidates[i] for i in idx]
            f_sorted = [f_vals[i] for i in idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Compute mean of mu best
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * y_sorted[i]
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv := np.linalg.cholesky(C).T @ (y_mean - x) / sigma)
            
            hsig = 1 if np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lambda_ + 1))) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma
            
            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(y_sorted[i] - x, y_sorted[i] - x) for i in range(mu))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            x = y_mean
            
            # If we've made good progress, we can stop early if sigma is very small
            if sigma < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
