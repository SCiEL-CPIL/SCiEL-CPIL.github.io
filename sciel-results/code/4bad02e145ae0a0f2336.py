# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 15  # Population size
        mu = lam // 2  # Number of best individuals for recombination
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = 1 / (dim + 3)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((dim + 2) / (dim + 4) - 1))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / dim) - 1) + c_sigma
        
        # Generation loop
        while calls_made < budget:
            # Sample population
            pop = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = np.random.standard_normal(dim)
                x = x_mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate population
            fvals = []
            for x in pop:
                f = func(x)
                fvals.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if calls_made >= budget:
                break
            
            # Sort
            indices = np.argsort(fvals)
            pop = [pop[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Recombination
            x_new_mean = sum(weights[i] * pop[i] for i in range(mu))
            
            # Update step size and covariance
            # Calculate C_inv_sqrt
            w_eig, V_eig = np.linalg.eigh(C)
            w_eig = np.maximum(w_eig, 1e-20)
            C_inv_sqrt = V_eig @ np.diag(1.0 / np.sqrt(w_eig)) @ V_eig.T
            
            diff = x_new_mean - x_mean
            d = np.linalg.norm(C_inv_sqrt @ diff)
            
            if d > 1e-12:
                # hsig
                hsig = 1.0 if np.linalg.norm(C_inv_sqrt @ diff) < (1.34 + 2.0 / (dim + 1)) * chi_n else 0.0
                
                p_sigma = (1 - c_sigma) * p_sigma + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / d * (C_inv_sqrt @ diff)
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # hpc
                # Check for drift
                # Simplified check
                hpc = 1.0 if np.linalg.norm(p_c) < (1.4 + 2.0 / (dim + 1)) * chi_n else 0.0
                
                p_c = (1 - c_c) * p_c + hpc * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                
                # Update C
                y_list = [C_inv_sqrt @ (pop[i] - x_mean) for i in range(mu)]
                rank_mu_update = sum(weights[i] * np.outer(y_list[i], y_list[i]) for i in range(mu))
                
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu_update
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    sigma = 1.0
                    x_mean = np.zeros(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    continue
            else:
                # No movement, maybe reset or continue
                pass

            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
