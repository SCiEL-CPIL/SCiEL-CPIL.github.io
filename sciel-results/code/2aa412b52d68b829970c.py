"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for Rastrigin-rotated function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Evaluate several diverse starting points to find a good basin
        rng = np.random.default_rng(42)
        candidates = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
        ]
        
        best_x = None
        best_f = np.inf
        for c in candidates:
            c = np.clip(c, -5, 5)
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()
        
        # CMA-ES parameters
        x = best_x.copy()
        f_best = best_f
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        calls_made = len(candidates)
        remaining = budget - calls_made
        
        if remaining <= 0:
            return x, f_best
        
        # CMA-ES loop
        while calls_made < budget:
            # Generate population
            offspring = []
            for i in range(min(lambda_, budget - calls_made)):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_made += 1
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x = offspring[0][1].copy()
            
            # Update mean
            y_w = np.array([offspring[i][1] for i in range(mu)])
            y_w -= x
            x_new = x + sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) < (1 + 2 / dim) * np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 / (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c) + c_mu / (1 - (1 - c_c) ** 2) * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_w, y_w), axis=0)
            
            # Renormalize covariance
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            x = x_new
            
            if calls_made >= budget:
                break
        
        return x, f_best
# EVOLVE-BLOCK-END
