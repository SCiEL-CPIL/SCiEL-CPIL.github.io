"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Population-based CMA-ES with adaptive restarts and error repair."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x_global = None
        best_f_global = np.inf
        
        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial mean - start at origin for symmetry
        x = np.zeros(dim)
        
        # Initial step size
        # Heuristic: 1/3 of the box width is a reasonable start for [-5, 5]
        sigma = 10.0 / 3.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        c_sigma = (lam/2 + 1) / (dim + 2 + lam/2)
        mu = lam // 2  # Number of selected individuals
        
        # Weights for selection
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Chi-square correction
        chi_lam = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        def run_cma_es(func, x0, f0, budget_remaining, max_iters, time_limit, start_time):
            """
            Runs CMA-ES starting from x0 with value f0.
            Returns (best_x, best_f, calls_made).
            """
            dim = len(x0)
            x = x0.copy()
            sigma = sigma_init  # Use global sigma_init
            C = np.eye(dim)
            C_eigen = np.eye(dim)
            C_eigenvalues = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Pre-compute eigen decomposition for efficient sampling
            def sample_normal():
                """Sample from N(0, C) using eigen decomposition."""
                z = np.random.randn(dim)
                # Transform using eigen decomposition
                return np.sqrt(C_eigenvalues) * (C_eigen @ z)
            
            best_x_so_far = x0.copy()
            best_f_so_far = f0
            calls_made = 1  # Already evaluated f0
            
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                if budget_remaining <= 0:
                    break
                
                # Generate population
                offspring = []
                offspring_f = []
                
                for i in range(lam):
                    # Sample from multivariate normal
                    z = sample_normal()
                    x_i = x + sigma * z
                    # Clip to bounds
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls_made += 1
                    budget_remaining -= 1
                    offspring.append(x_i)
                    offspring_f.append(f_i)
                    
                    if budget_remaining <= 0:
                        break
                
                if budget_remaining <= 0:
                    break
                
                # Sort offspring by fitness
                offspring_f = np.array(offspring_f)
                offspring = np.array(offspring)
                sort_idx = np.argsort(offspring_f)
                offspring = offspring[sort_idx]
                offspring_f = offspring_f[sort_idx]
                
                # Update best
                if offspring_f[0] < best_f_so_far:
                    best_f_so_far = offspring_f[0]
                    best_x_so_far = offspring[0].copy()
                
                # Update mean
                x_new = np.sum(weights[:, np.newaxis] * offspring[:mu], axis=0)
                
                # Update covariance matrix
                # First, compute the difference vectors
                y = (x_new - x) / sigma
                y_weighted = np.sqrt(mu_eff) * y
                
                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * y_weighted
                p_c_norm = np.linalg.norm(p_c)
                
                # Update p_sigma
                h_sigma = 1 if p_c_norm / np.sqrt(1 - (1 - c_c)**2) < (1 + 2/dim) * chi_lam else 0
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma)) * y_weighted
                
                # Update C
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(y_weighted, y_weighted)
                
                # Also add rank-mu update
                C += c_mu * np.sum(weights[:mu] * np.outer(offspring[:mu] - x, offspring[:mu] - x) / sigma**2, axis=0)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_lam * (np.linalg.norm(p_sigma) / chi_lam - 1))
                
                # Update x
                x = x_new
                
                # Periodically recompute eigen decomposition
                # This is expensive, so do it every 10 iterations
                if _ % 10 == 0:
                    C_eigenvalues, C_eigen = np.linalg.eigh(C)
                    # Ensure positive eigenvalues
                    C_eigenvalues = np.maximum(C_eigenvalues, 1e-10)
                
                # Stagnation check: if sigma is too small or no improvement
                if sigma < 1e-8 or np.std(offspring_f[:mu]) < 1e-10:
                    break
                
                # Error repair: if we're stuck, try a random restart from best
                if _ > 50 and offspring_f[0] >= best_f_so_far:
                    # Try a small perturbation of the best
                    x_perturbed = best_x_so_far + sigma * 0.1 * np.random.randn(dim)
                    x_perturbed = np.clip(x_perturbed, -5, 5)
                    f_perturbed = func(x_perturbed)
                    calls_made += 1
                    budget_remaining -= 1
                    if f_perturbed < best_f_so_far:
                        best_f_so_far = f_perturbed
                        best_x_so_far = x_perturbed.copy()
                        x = x_perturbed.copy()
                        sigma = max(sigma, 0.5)
                    
                    if budget_remaining <= 0:
                        break
        
            return best_x_so_far, best_f_so_far, calls_made

        # Main loop: multi-start CMA-ES
        # Phase 1: Initial diverse sampling
        n_init_samples = max(5, min(15, budget // 20))
        
        candidates = []
        for i in range(n_init_samples):
            if time.time() - start_time > time_limit:
                break
            if budget <= 0:
                break
            x = np.random.uniform(-4, 4, dim)
            f = func(x)
            budget -= 1
            candidates.append((f, x))
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Run CMA-ES from best candidates
        processed_starts = 0
        max_restarts = budget // 30  # CMA-ES is more expensive than NM
        
        while budget > 0 and time.time() - start_time < time_limit:
            # Get next start point
            if processed_starts < len(candidates):
                f_start, x_start = candidates[processed_starts]
                processed_starts += 1
            else:
                # Generate new random start
                x_start = np.random.uniform(-4, 4, dim)
                f_start = func(x_start)
                budget -= 1
                if f_start < best_f_global:
                    best_f_global = f_start
                    best_x_global = x_start.copy()
                
                if budget <= 0:
                    break
            
            # Run CMA-ES
            x_best_local, f_best_local, calls_used = run_cma_es(
                func, x_start, f_start, budget, 100, time_limit, start_time
            )
            
            budget -= calls_used
            
            if f_best_local < best_f_global:
                best_f_global = f_best_local
                best_x_global = x_best_local.copy()
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
