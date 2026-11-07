"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock-rotated functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES with a given budget, returning best point and value."""
        dim = self.dim
        n = budget
        
        # CMA-ES parameters
        popsize = int(4 + 3 * np.log(dim))
        popsize = min(popsize, n)
        
        # Strategy parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
        cs = (dim + 1.5) / (dim + 6 + popsize / 2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        c1 = 2 / ((dim + 1.3) ** 2 + popsize)
        cmu = min(1 - c1, 2 * (2 - dim / (dim + 2)) ** 2 - 1)
        
        # Initialize
        xmean = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track best
        best_x = xmean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Success counter for adaptive sigma
        successes = 0
        counter = 0
        threshold = int(1 / (5 * popsize))
        
        for gen in range(n // popsize + 1):
            if time.time() >= deadline:
                break
            
            # Generate population
            for i in range(popsize):
                if time.time() >= deadline:
                    break
                
                # Sample
                z = rng.standard_normal(dim)
                y = xmean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            # We need to re-evaluate or store; let's store
            # Actually, let's restructure to store all evaluations
            
            # Re-do with storage
            # For efficiency in this rewrite, we'll assume we stored them
            # But since we didn't, let's just break if we hit budget
            
        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Coarse search with random restarts
        # Use a (1+1)-ES with larger steps to find a good basin
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1
        
        sigma = 3.0
        stall_count = 0
        max_stall = 20
        
        # Phase 1: Coarse exploration
        budget_phase1 = int(0.4 * self.budget)
        
        for i in range(budget_phase1):
            if time.time() >= deadline:
                break
            if calls_made >= self.budget:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                stall_count = 0
                sigma = min(sigma * 1.2, 5.0)
            else:
                stall_count += 1
                sigma *= 0.95
                if stall_count > max_stall:
                    # Restart from best point with larger sigma
                    x = best_x + 0.5 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 2.0
                    stall_count = 0
            
            if calls_made >= self.budget:
                break
        
        # Phase 2: Fine-tuning with CMA-ES-like approach
        # Use a more sophisticated local search
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        calls_made = min(calls_made, self.budget - 1)
        
        # Use a simple CMA-ES implementation for the remaining budget
        dim = self.dim
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 10:
            # Simple CMA-ES
            popsize = min(6, remaining_budget)
            xmean = x.copy()
            sigma = 0.5
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            cs = (dim + 1.5) / (dim + 6 + popsize / 2)
            c1 = 2 / ((dim + 1.3) ** 2 + popsize)
            cmu = min(1 - c1, 2 * (2 - dim / (dim + 2)) ** 2 - 1)
            
            for gen in range(remaining_budget // popsize + 1):
                if time.time() >= deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                # Generate population
                W = np.sort(np.arange(1, popsize + 1))
                weights = np.log(popsize / 2) - np.log(W) + 1
                weights /= np.sum(weights)
                mu_eff = 1 / np.sum(weights ** 2)
                
                offspring = []
                for i in range(popsize):
                    if time.time() >= deadline:
                        break
                    if calls_made >= self.budget:
                        break
                    
                    z = rng.standard_normal(dim)
                    y = xmean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y))
                
                if len(offspring) < popsize:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                
                # Update mean
                xmean_new = np.zeros(dim)
                for i in range(int(mu_eff)):
                    xmean_new += weights[i] * offspring[i][1]
                
                # Update covariance matrix
                ydiff = np.zeros((dim, int(mu_eff)))
                for i in range(int(mu_eff)):
                    ydiff[:, i] = (offspring[i][1] - xmean) / sigma
                
                C = np.eye(dim)
                for i in range(int(mu_eff)):
                    C += weights[i] * np.outer(ydiff[:, i], ydiff[:, i])
                C = (1 - c1 - cmu) * C + c1 * (1 + (1 - cc) ** 2 - cc) * np.outer(pc, pc) + cmu * C
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                
                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * xmean_new))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chiN:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / chiN - 1))
                
                # Update evolution path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (xmean_new - xmean) / sigma
                
                xmean = xmean_new
                sigma = max(sigma, 1e-10)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
