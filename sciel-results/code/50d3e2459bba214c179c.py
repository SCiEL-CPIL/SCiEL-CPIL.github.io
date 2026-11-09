"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with time-based budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        # CMA-ES parameters
        dim = self.dim
        popsize = 4 + int(3 * np.log(dim))
        popsize = max(popsize, 10)
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, roughly 1/4 of box width
        
        # CMA-ES state
        C = np.eye(dim)
        Cs = np.zeros(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Adaptation parameters
        mu = popsize // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial population
        x_best = None
        f_best = np.inf
        
        # We need to ensure we don't exceed the budget
        # But since the deadline is the primary constraint for complex cases,
        # we loop until time runs out or budget is exhausted.
        # However, the contract says "More than 500 calls... makes the whole result invalid".
        # So we MUST strictly limit to 500 calls.
        
        calls_made = 0
        max_calls = self.budget
        
        while calls_made < max_calls:
            if time.time() > deadline:
                break
                
            # Sample popsize individuals
            # We need to be careful not to exceed max_calls in a single batch
            # If remaining budget < popsize, we reduce popsize for this iteration
            current_popsize = min(popsize, max_calls - calls_made)
            if current_popsize < 2:
                # Not enough budget for a proper population step, do a final single step or break
                if current_popsize == 1:
                    y = np.clip(mean + sigma * np.random.randn(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                break

            # Generate population
            B = np.linalg.cholesky(C)
            z = np.random.randn(current_popsize, dim)
            # Whitened and transformed
            Y = mean + sigma * (z @ B.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += current_popsize
            
            # Track best
            idx_best = np.argmin(F)
            if F[idx_best] < f_best:
                f_best = F[idx_best]
                x_best = Y[idx_best]
                
            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update path cumulation (Cs)
            Cs = (1 - c_sigma) * Cs + c_sigma * (new_mean - mean) / sigma
            
            # Update path cumulation (pc)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            
            # Update covariance matrix
            # h_sigma = 1 if ||Cs|| / sqrt(1 - (1-c_sigma)^{2*(calls_made+1)}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+1)/(mu_eff-1)) * sqrt(1 - (1-c_sigma)^{2*(calls_made+1)})
            # Simplified check
            norm_cs = np.linalg.norm(Cs)
            threshold = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt((mu_eff + 1.0) / (mu_eff - 1.0))
            h_sigma = 1.0 if norm_cs / np.sqrt(1.0 - (1.0 - c_sigma)**(2.0 * (calls_made + 1.0))) < threshold else 0.0
            
            # Rank-one update
            rank1 = np.outer(pc, pc)
            # Rank-mu update
            v = (Y[:mu] - mean) / sigma
            rankmu = np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->jk', v, v), axis=0)
            
            C = (1.0 + c_1 * h_sigma * (np.outer(pc, pc) - C) + c_mu * (rankmu - C)) * (1.0 - c_1 * h_sigma)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_cs / np.sqrt(mu_eff) - 1.0))
            
            # Update mean
            mean = new_mean
            
            # Project mean back to box? No, CMA-ES works in unbounded space usually, 
            # but we clip the samples. The mean can go outside, but samples are clipped.
            # However, if mean goes far outside, it might be inefficient. 
            # For this task, clipping is sufficient.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
