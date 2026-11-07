"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        dim = self.dim
        n = self.budget

        # CMA-ES parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))
        # For dim=5, pop_size = 4 + floor(3*1.609) = 4 + 4 = 8
        
        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        
        # CMA-ES parameters
        c_sigma = (pop_size + 2) / ((dim + 1.3)**2 + pop_size)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3)**2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2)**2 + pop_size))
        
        # Weights for selection
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        
        # Precompute Cholesky factor
        C_cholesky = np.linalg.cholesky(C)
        
        # Number of evaluations per generation
        evals_per_gen = pop_size
        
        # Main loop
        for gen in range(n // evals_per_gen + 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            fys = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * C_cholesky @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fys.append(fy)
            
            if len(ys) < pop_size:
                # If we couldn't generate all offspring due to time, use what we have
                if len(ys) == 0:
                    break
                # Pad with current best if needed (though this shouldn't happen often)
                while len(ys) < pop_size:
                    ys.append(best_x)
                    fys.append(best_fx)
            
            # Sort by fitness
            idx = np.argsort(fys)
            ys = [ys[i] for i in idx]
            fys = [fys[i] for i in idx]
            
            # Update best
            if fys[0] < best_fx:
                best_fx = fys[0]
                best_x = ys[0].copy()
            
            # Compute mean of selected points
            y_w = sum(weights[i] * ys[i] for i in range(pop_size))
            
            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_cholesky.T @ (y_w - x) / sigma)
            
            # Update covariance matrix adaptation
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) if gen > 0 else 0
            # Simplified h_sigma check
            h_sigma = 1 if np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - (1 - c_sigma)**(2 * (gen + 1))) / (1 - (1 - c_sigma)**(2 * (gen + 1 + pop_size)))) else 0
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (y_w - x) / sigma
            
            # Rank-1 and rank-mu update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            diff = np.array([ys[i] - y_w for i in range(pop_size)])
            C += c_mu * np.sum(weights[i] * np.outer(diff[i], diff[i]) for i in range(pop_size))
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = y_w
            
            # Update Cholesky factor
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, reset
                C = np.eye(dim)
                C_cholesky = np.eye(dim)
                sigma = max(sigma, 0.1)
            
            # Check convergence
            if np.all(np.abs(sigma * C_cholesky) < 1e-10):
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
