"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lambda_ = max(4 * dim + 4, 10)
        mu_ = lambda_ // 2
        
        # Weights
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial exploration: sample a few diverse points to locate the basin
        # This helps if the optimum is far from the origin
        init_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
        ]
        # Add axis-aligned points
        for i in range(dim):
            for val in [2.0, -2.0]:
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)
        
        # Evaluate initial points (limited to not waste too much budget)
        max_init = min(len(init_points), max(5, budget // 10))
        for pt in init_points[:max_init]:
            if call_count >= budget - lambda_:
                break
            if time.time() > deadline:
                break
            _evaluate(pt)
        
        # Set initial mean to the best point found
        if best_x is not None:
            x_mean = best_x.copy()
        else:
            x_mean = np.zeros(dim)
        
        # Initial sigma: use a moderate value to allow exploration
        sigma = 1.5
        
        # Covariance matrix C
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Chi_n constant for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                f_y = _evaluate(y)
                offspring.append((y, f_y, z))
            
            if len(offspring) < mu_:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu_]
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)
            
            # Update evolution paths
            C_inv_sqrt = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
            delta_mean = x_mean - old_mean
            
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Update sigma
            z_mean = np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z_mean
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance Matrix
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * np.eye(dim))
            
            for i in range(mu_):
                diff = selected[i][0] - x_mean
                C += c_mu * w[i] * np.outer(diff, diff)
            
            C = (C + C.T) / 2  # Symmetrize
            
            # Ensure positive semi-definite
            try:
                eigvals_C, _ = np.linalg.eigh(C)
                if np.min(eigvals_C) < 1e-12:
                    C = C + (1e-10 - np.min(eigvals_C)) * np.eye(dim)
            except:
                C = np.eye(dim)
                
            # Update eigen decomposition for next iteration
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-12))
                B = eigvecs
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
