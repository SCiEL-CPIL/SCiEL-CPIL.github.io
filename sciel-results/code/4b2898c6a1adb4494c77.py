"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        start_time = time.time()
        max_time = 290.0 # Safety margin against 300s limit
        
        # CMA-ES Parameters
        # For dim=5, standard lambda is 4 + floor(3*log(5)) = 4+4 = 8. 
        # We use a slightly larger population for better exploration in the first phase.
        lam = 12
        mu = lam // 2
        
        # Initial step size
        sigma = 1.0
        
        # We will perform a multi-start strategy.
        # Phase 1: Quick exploration with several random starts to find promising basins.
        # Phase 2: Refinement of the best basin found.
        
        # Given the tight budget (500 calls), we must be efficient.
        # Let's split the budget: 
        # ~100 calls for initial multi-start screening (2 starts of ~50 calls each? or 1 start of 100?)
        # Actually, CMA-ES converges fast. 
        # Strategy: 
        # 1. Start CMA-ES at origin (0,0,0,0,0) or random? Lunacek has a shifted global min.
        #    Random start is safer.
        # 2. Run CMA-ES until budget is exhausted or convergence.
        # 3. If time/budget remains, restart CMA-ES from the best point found with a smaller sigma?
        #    Or better: Just run one robust CMA-ES instance with a good initial population.
        
        # Let's use a single robust CMA-ES run but with a slightly larger initial sigma 
        # and a mechanism to detect stagnation and restart if necessary.
        
        # However, the "Multi-Start" insight suggests that sampling multiple basins is key.
        # Let's do 3 distinct CMA-ES runs sequentially if budget allows? 
        # 500 / 3 = 166 calls per run. That might be too few for CMA-ES to fully converge.
        
        # Alternative: One CMA-ES run, but initialize the mean with a random point 
        # and let it adapt. CMA-ES is very robust.
        
        # Let's try a hybrid: 
        # 1. Use a small population CMA-ES.
        # 2. If no improvement for X generations, restart the search distribution 
        #    centered at the current best point with a larger sigma (or smaller if stuck in a peak? no, minimization).
        
        # Actually, the most "structurally different" and powerful approach for Lunacek 
        # is often just a well-tuned CMA-ES. The rotation is handled by the covariance matrix.
        
        # Implementation of CMA-ES core loop
        
        def cma_es_run(start_x, sigma_init, budget_for_run, rng):
            """
            Runs CMA-ES for a specific number of function calls.
            Returns best_x, best_f, calls_used
            """
            # Initialize
            mean = np.array(start_x, dtype=float)
            sigma = sigma_init
            C = np.eye(dim)
            C_inv = np.eye(dim)
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Parameters
            # Standard CMA-ES parameters
            c_sigma = 1.0 / (4 * dim + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_mu = (2 + 2 * mu / dim) / (dim + 6 + 2 * mu / dim)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            calls_used = 0
            best_x = mean.copy()
            best_f = np.inf
            
            while calls_used + lam <= budget_for_run:
                if time.time() - start_time > max_time:
                    break
                
                # Generate offspring
                # Sample from N(mean, sigma^2 * C)
                # Cholesky decomposition of C for sampling
                L = np.linalg.cholesky(C)
                z = rng.standard_normal((lam, dim))
                y = z @ L.T # (lam, dim)
                x_offspring = mean + sigma * y
                x_offspring = np.clip(x_offspring, -5, 5)
                
                # Evaluate
                fvals = np.array([func(x) for x in x_offspring])
                calls_used += lam
                
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = x_offspring[min_idx].copy()
                
                # Sort offspring
                order = np.argsort(fvals)
                x_offspring = x_offspring[order]
                fvals = fvals[order]
                
                # Update mean
                w = weights
                new_mean = np.sum(w[:, None] * x_offspring[:mu], axis=0)
                
                # Update covariance matrix
                # y_k = (x_k - mean) / sigma
                y_k = (x_offspring[:mu] - mean) / sigma
                
                # Delta rule
                delta = (new_mean - mean) / sigma
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * w @ y_k
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
                
                # Heaviside
                h_sig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lam + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim else 0.0
                h_c = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (calls_used // lam + 1))) < (1.4 + c_c - 1/c_c) * np.trace(C) / dim else 0.0
                
                # Update C
                C = (1 - c_mu) * C + c_mu * (np.outer(delta, delta) + (1 - h_c) * c_c * (2 - c_c) * C - p_c @ p_c.T)
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2.0
                # Regularize if necessary
                # For dim=5, this is usually stable.
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update mean
                mean = new_mean
                
                # Convergence check: if sigma is very small or variance is tiny
                if sigma < 1e-8:
                    break
                
                # Optional: restart logic if stuck? 
                # For now, rely on the single run robustness.
                
            return best_x, best_f, calls_used

        # Strategy: 
        # Use the full budget for one CMA-ES run, but initialize with a random point 
        # to avoid biasing towards the origin if the shift is large.
        # However, Lunacek's global minimum is often near the origin in the transformed space, 
        # but the task says "hidden seeded shift". 
        # Random start is safer.
        
        rng = np.random.default_rng(42)
        
        # Start at a random point in [-5, 5]^dim
        start_x = rng.uniform(-5, 5, size=dim)
        
        best_x, best_f, used_calls = cma_es_run(start_x, sigma_init=1.0, budget_for_run=total_budget, rng=rng)
        
        # If we have budget left (e.g., due to early break or loop condition), 
        # we could try a second start, but CMA-ES usually uses all or breaks on convergence.
        # Given the 500 budget and lam=12, we get ~41 generations.
        # This is reasonable.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
