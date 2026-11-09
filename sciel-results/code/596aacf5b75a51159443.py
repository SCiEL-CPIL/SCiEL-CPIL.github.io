"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda is the population size
        lam = 10 
        mu = lam // 2 # number of selected individuals
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES specific parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        rng = np.random.default_rng(0)
        x = np.zeros(dim) # Start at origin, usually a good heuristic for centered boxes
        sigma = 2.0 # Initial step size
        
        # State variables
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1/(2*dim) - 1/(4*dim**2))
        
        # Evaluation tracking
        calls = 0
        best_x = x.copy()
        best_f = func(x)
        calls += 1
        
        start_time = time.time()
        
        while calls < budget:
            # Time guard: ensure we have at least 1 second left to finish safely
            if time.time() - start_time > 325:
                break
                
            # Generate population
            population = []
            for i in range(lam):
                # Generate standard normal
                z = rng.standard_normal(dim)
                # Apply transformation
                y = x + sigma * (B @ (D * z))
                # Clip to bounds [-5, 5]
                y = np.clip(y, -5, 5)
                population.append(y)
            
            # Evaluate population
            values = []
            for y in population:
                if calls >= budget:
                    break
                fy = func(y)
                values.append(fy)
                calls += 1
                
            if len(values) < mu:
                break
                
            # Sort by fitness
            sorted_idx = np.argsort(values)
            selected_idx = sorted_idx[:mu]
            selected_y = [population[i] for i in selected_idx]
            selected_f = [values[i] for i in selected_idx]
            
            # Update best
            if selected_f[0] < best_f:
                best_f = selected_f[0]
                best_x = selected_y[0].copy()
                
            # Recombination
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += w[i] * selected_y[i]
                
            # Update step size control variables
            y_vec = (new_x - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n) < (1.4 + 2 / (dim + 1))
            
            # Update covariance matrix adaptation
            pc = (1 - c_c) * pc + hsig * c_c * y_vec
            
            # Rank-one and rank-mu updates
            C_update = (1 + c_1 * (2 - c_1) * hsig - c_1) * np.eye(dim)
            C_update += c_1 * np.outer(pc, pc)
            
            for i in range(mu):
                y_i = (selected_y[i] - x) / sigma
                C_update += c_mu * w[i] * np.outer(y_i, y_i)
                
            # Update B and D
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_update)
            # Ensure positive definiteness
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = new_x
            
            # Optional: If sigma becomes too small or too large, we might stall.
            # But CMA-ES is robust.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
