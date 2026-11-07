"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with fixed step size schedule and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(0)
        
        # Initial best point
        x_best = np.zeros(self.dim)
        f_best = func(x_best)
        
        calls_made = 1
        lambda_pop = 10
        
        # CMA-ES parameters
        # Standard CMA-ES initialization
        sigma = 2.0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Step size parameters
        c_sigma = (lambda_pop + 2) / (self.dim + lambda_pop + 5)
        mu_sigma = (lambda_pop + 2) / (self.dim + lambda_pop + 5)
        
        # Mean update parameters
        c_c = 4 / (self.dim + 12)
        c_1 = 2 / ((self.dim + 1.3)**2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((lambda_pop/2 + 1) - self.dim - 2) / ((self.dim + 2)**2 + lambda_pop))
        
        # Weights
        w = np.log(np.arange(1, lambda_pop + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Chi-squared expectation
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(32*self.dim**2))
        
        # Schedule for sigma: start large, decay
        # We will manually adjust sigma based on progress to avoid complex adaptation 
        # that might be too slow for 500 calls, but keep C for rotation.
        # Actually, let's use a simplified CMA-ES where we update C but fix sigma 
        # or update it very simply.
        
        # Let's stick to a robust implementation:
        # We will run CMA-ES for a few generations, then restart.
        
        gen_count = 0
        no_improve = 0
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Use Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)
            
            pop = []
            f_pop = []
            
            # We will generate lambda_pop - 1 new points, keeping x_best as one?
            # Standard CMA-ES generates lambda_pop new points from mean x.
            # Here x_best is our current mean candidate.
            
            for i in range(lambda_pop):
                z = rng.standard_normal(self.dim)
                y = x_best + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                pop.append(y)
                f_pop.append(fy)
                
                if fy < f_best:
                    f_best = fy
                    x_best = y
            
            # Sort population
            indices = np.argsort(f_pop)
            pop = np.array(pop)
            f_pop = np.array(f_pop)
            pop = pop[indices]
            f_pop = f_pop[indices]
            
            # Update mean
            mu = int(np.floor(lambda_pop / 2))
            x_new = np.sum(w[:mu, np.newaxis] * pop[:mu], axis=0)
            
            # Update covariance
            y_inv = (pop[:mu] - x_best) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (c_c * (2 - c_c)) * np.eye(self.dim)) + c_mu * np.sum(w[:mu, np.newaxis, np.newaxis] * np.outer(y_inv, y_inv), axis=0)
            
            # Update path states
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.norm(y_inv.mean(axis=0, keepdims=True)) / chi_n * (y_inv.mean(axis=0) / (sigma * np.linalg.norm(y_inv.mean(axis=0)) + 1e-10) if np.linalg.norm(y_inv.mean(axis=0)) > 1e-10 else 0)
            # Simplified p_sigma update for stability:
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.linalg.norm(y_inv.mean(axis=0)) / chi_n) * (y_inv.mean(axis=0) / (np.linalg.norm(y_inv.mean(axis=0)) + 1e-10))
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_inv.mean(axis=0) / sigma)
            
            x_best = x_new
            
            # Adapt sigma
            # Standard CMA-ES sigma update
            # sigma *= exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            # To keep it simple and robust within budget, we might just decay sigma slightly
            # or use the standard rule. Let's use standard rule but clip it.
            sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            sigma = np.clip(sigma, 1e-5, 10.0)
            
            # Check for stagnation
            if f_best > f_pop[0] * 1.0001: # No improvement over previous best
                no_improve += 1
            else:
                no_improve = 0
            
            # Restart if stagnated or sigma too small
            if no_improve > 5 or sigma < 0.01:
                # Restart from current best with fresh C and larger sigma
                C = np.eye(self.dim)
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)
                sigma = 1.0
                no_improve = 0
                # Keep x_best as is, it's the best found so far
                # Force a jump? No, just restart the local search around it.
                # To explore more, we could add a random jump, but that risks losing progress.
                # Let's just restart the local CMA-ES.
            
            gen_count += 1
            if gen_count > 50: # Safety break for infinite loops
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
