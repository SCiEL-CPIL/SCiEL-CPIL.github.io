"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Single-run CMA-ES with aggressive convergence settings."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def check_deadline():
            return (time.time() - start_time) > deadline

        # CMA-ES Parameters
        # We use a relatively large population to ensure robust exploration
        # but not so large that it wastes budget on redundant evaluations.
        # For dim=5, lambda=4*int(dim/2+1) = 4*3 = 12. 
        # Let's try a slightly larger pop size for better gradient estimation early on.
        pop_size = 4 * int(dim / 2 + 1)
        pop_size = max(pop_size, 10) # Ensure at least 10
        
        # Standard CMA-ES parameters
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.arange(mu)
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cs = (dim + 1.7) / (dim + 22)
        c1 = 2 / ((dim + 1.3) ** 2 + 10)
        cmu = min(0.3 - c1, (2 - c1 + 2 / mu_eff) / ((dim + 2) ** 2 - 1))
        
        # Initialize state
        # Start at the origin, which is a strong prior for BBOB functions
        x_mean = np.zeros(dim)
        f_mean = np.inf
        
        # Initial evaluation
        f_mean = call_func(x_mean)
        
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Precompute Cholesky factor update logic
        # We will update C and recompute Cholesky each iteration for simplicity and robustness
        
        it = 0
        while calls_made < budget and not check_deadline():
            # Generate population
            offspring = []
            offspring_f = []
            
            for k in range(pop_size):
                if calls_made >= budget or check_deadline():
                    break
                z = np.random.standard_normal(dim)
                # Use current Cholesky factor
                L = np.linalg.cholesky(C)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]
            
            # Update mean
            x_mean_new = np.sum(w[:, None] * offspring[:mu], axis=0)
            
            # Update path cumulation
            pc = (1 - cc) * pc + cc * (1/sigma) * (x_mean_new - x_mean)
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1-cc)**(2*(it+1)))) / chi_n < (1.4 + 2/(dim+1))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * hsig) * (1/sigma) * (x_mean_new - x_mean)
            
            # Update covariance matrix
            delta = (1 + hsig * cs * (2 - cs)) * c1 * (np.outer(pc, pc) - C)
            rank_mu = (offspring[:mu] - x_mean)
            delta += cmu * np.sum(w[:, None, None] * np.einsum('i,ij,ik->ijk', w, rank_mu, rank_mu), axis=0)
            C = (1 + c1 + cmu - c1) * C + delta
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if necessary
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity to fix PD
                C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # Update sigma
            sigma *= np.exp(1.0 / (1 + 2 * max(0, np.sqrt(dim) - 1) / (dim + 3)) * (np.linalg.norm(ps)/chi_n - 1))
            
            # Update mean
            x_mean = x_mean_new
            f_mean = offspring_f[0]
            
            it += 1
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if sigma > 1e6:
                break
            # If we have converged significantly, we can stop early to save time, 
            # but the budget is small enough that running full is usually fine.
            # However, if progress is nil, we might restart? 
            # The directive says "single, robust... run", so we stick to one run.
            # But if we stall, we can break.
            if it % 10 == 0:
                # Check if we are stuck
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
