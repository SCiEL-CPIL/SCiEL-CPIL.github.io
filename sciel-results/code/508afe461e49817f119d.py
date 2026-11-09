"""CMA-ES based optimizer for BBOB f13."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """CMA-ES with restarts for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(0)
        
        # Track the global best found so far
        best_x = None
        best_f = np.inf

        # Initial population for the first restart
        # Use a broad initial sigma to ensure global coverage
        sigma0 = 2.0
        
        # We will run multiple restarts of CMA-ES.
        # Each restart gets a portion of the budget.
        # A simple strategy: allocate budget evenly or adaptively.
        # Given the "sharp ridge" nature, we want fast convergence.
        
        # Number of restarts: Let's try 5 restarts.
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES parameters
            # Center: For the first restart, center at 0. For subsequent, 
            # we can center near the best found so far or random.
            # Random restarts are generally more robust for global opt.
            if r == 0:
                x0 = np.zeros(self.dim)
            else:
                # Random restart within the box
                x0 = rng.uniform(-4.5, 4.5, self.dim)
            
            # CMA-ES parameters
            # lambda = 4 * ceil(3 + log(dim)) is a common choice
            lam = 4 * int(np.ceil(3 + np.log(self.dim)))
            # Ensure lam is not too small
            lam = max(lam, 4)
            
            # We implement a simplified CMA-ES loop to control the budget exactly
            # and handle the time constraint.
            
            # CMA-ES State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            # Adaptation parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c1 = (2 / (self.dim + 1.3)**2 + 2 / (self.dim + mu_eff - 2)) / ((self.dim + 2) * (2 / (self.dim + 1.3)**2 + 2 * (1 - 1 / mu_eff) / (self.dim + 1.3)**2))
            cmu = min(1 - c1, 2 * (weights[0] - 1 + 1 / mu_eff) / ((self.dim + 2) * (1 + 1 / mu_eff)))
            
            # Initial population
            # We need to evaluate 'lam' points per generation.
            # Total budget is 500.
            # Generations = budget // lam
            max_generations = budget_per_restart // lam
            
            # If lam is large, generations are small. 
            # For dim=5, lam = 4 * ceil(3 + 1.6) = 4 * 5 = 20.
            # 500 / 20 = 25 generations per restart.
            # 5 restarts * 25 gens = 125 generations total.
            # This seems reasonable.
            
            for gen in range(max_generations):
                if time.time() > deadline:
                    break
                
                # Sample population
                z = rng.standard_normal((lam, self.dim))
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(self.dim)
                
                pop = mean + sigma * (z @ L.T)
                
                # Evaluate population
                fitness = np.array([func(p) for p in pop])
                
                # Sort by fitness
                order = np.argsort(fitness)
                best_idx = order[0]
                
                # Update global best
                if fitness[best_idx] < best_f:
                    best_f = fitness[best_idx]
                    best_x = pop[best_idx].copy()
                    
                # Select top mu
                selected = pop[order[:mu]]
                selected_z = z[order[:mu]]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * selected, axis=0)
                
                # Update step size control
                y_mean = (new_mean - mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ y_mean)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(gen + 1)) < 
                        (1 + 2 / (self.dim + 1)) * np.sqrt(self.dim + 1 - 2))
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * cc * y_mean
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu / sigma**2 * np.sum(weights[:, None, None] * np.einsum('ij,ik->jk', selected_z, selected_z), axis=0)
                
                # Update sigma
                sigma *= np.exp(cs / (self.dim + 1.3) * (np.linalg.norm(ps) / np.sqrt(2 * self.dim) - 1))
                
                # Update mean
                mean = new_mean
                
                # Clip mean to box? No, CMA-ES handles bounds via sampling or projection.
                # The task says func clips inputs. So we don't need to clip mean/pop explicitly 
                # for the optimizer logic, but it helps to keep mean in bounds for efficiency.
                # However, clipping mean can bias the distribution. 
                # Since func clips, we can just let it be.
                
                # Termination criteria:
                # 1. Sigma too small
                # 2. No progress
                # 3. Time limit
                if sigma < 1e-6:
                    break
                    
        return best_x, best_f

# EVOLVE-BLOCK-END
