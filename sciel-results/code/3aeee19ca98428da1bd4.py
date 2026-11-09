"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f11 (Discus) with wall-clock safety."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mus = lam // 2
        weights = np.log(mus + 0.5) - np.log(np.arange(1, mus + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cmacov = 1.0 / (4 + np.sqrt(lam / 2))
        cpc = 1.0 / (4 + dim / 2)
        cc = (4 + dim / 2) / (dim + 4)
        
        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Eigenvalue decomposition for C (initially identity)
        D_inv_sqrt = np.eye(dim)
        B = np.eye(dim)
        
        # Initial population
        population = []
        for i in range(lam):
            z = np.random.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            population.append((fy, y, z))
            if time.time() > deadline:
                return x, fx
                
        best_x = x.copy()
        best_f = fx
        for fy, y, z in population:
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                
        population.sort(key=lambda t: t[0])
        
        # Main loop
        for gen in range(budget - lam):
            if time.time() > deadline:
                break
                
            # 1. Update mean
            x = sum(weights[i] * population[i][1] for i in range(mus))
            
            # 2. Update evolution paths
            mean_z = sum(weights[i] * population[i][2] for i in range(mus))
            
            ps = (1 - cmacov) * ps + np.sqrt(cmacov * (2 - cmacov) * mu_eff) * mean_z
            
            # hsig calculation
            hsig = np.linalg.norm(ps) / (np.sqrt(1 - (1 - cmacov) ** (2 * (gen + 1))) * (1 + 2 / (dim + 1)) * np.sqrt(mu_eff))
            
            pc = (1 - cpc) * pc + hsig * cc * (B.T @ (D_inv_sqrt.T @ mean_z))
            
            # 3. Update Covariance
            hsig_c = hsig < (1.4 + 2 / (dim + 1))
            delta_hs = 1 - hsig_c
            
            # Rank-mu update
            C = (1 + cmacov * (2 - cmacov) + delta_hs * cc * (2 - cpc)) * C
            for i in range(mus):
                z_i = population[i][2]
                C -= cmacov * (C @ z_i[:, None] @ z_i[None, :] / cmacov + z_i[:, None] @ z_i[None, :] @ C / cmacov)
                C += cmacov * (2 - cmacov) * np.outer(z_i, z_i) * weights[i] / mu_eff
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigenvalue decomposition for next iteration
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
            
            # 4. Update sigma (CSA)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            ds = 1 + 2 * max(0, (dim - 1) / (dim + 3)) + cs
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # 5. Generate next population
            population = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = np.clip(x + sigma * (B @ (D_inv_sqrt @ z)), -5, 5)
                fy = func(y)
                population.append((fy, y, z))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if time.time() > deadline:
                    return best_x, best_f
                    
            population.sort(key=lambda t: t[0])
            
        return best_x, best_f
# EVOLVE-BLOCK-END
