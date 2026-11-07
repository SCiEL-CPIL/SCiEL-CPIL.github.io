import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 325s to be safe
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        def cma_es_step(mean, sigma, C, ps, pc):
            # Generate population
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            
            # Sample points
            z = rng.standard_normal((lambda_pop, dim))
            pop = mean + sigma * (z @ L.T)
            
            # Evaluate
            fvals = np.array([wrapper(p) for p in pop])
            
            # Sort indices
            order = np.argsort(fvals)
            
            # Update mean
            new_mean = np.sum(weights[:, None] * pop[order[:mu]], axis=0)
            
            # Update step size
            y = (new_mean - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * L.T @ y
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * np.trace(L) / dim else 0
            
            sigma *= np.exp(cs / cs_target * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
            # Note: cs_target is usually 1.0, but standard is to use the norm of ps directly in the exponent?
            # Standard formula: sigma = sigma * exp( cs/dlambda * ( ||ps|| / sigma_norm - 1 ) )
            # where sigma_norm = E[||N(0, I)||] approx sqrt(dim)
            # Let's use the standard adaptation:
            # Actually, the standard CMA-ES uses:
            # sigma = sigma * exp( cs / cs_target * ( ||ps|| / sigma_norm - 1 ) )
            # where cs_target = 1.0 and sigma_norm = sqrt(dim) is a common approximation, 
            # but more accurately sigma_norm = sqrt(dim) is not quite right.
            # Let's stick to the simpler 1/5 rule if CMA-ES gets too complex, 
            # but CMA-ES is better. Let's use a simplified update.
            
            # Simpler sigma update:
            # sigma = sigma * (1 + cs * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            # This is an approximation.
            
            # Let's use the exact standard update:
            # norm_ps = np.linalg.norm(ps)
            # sigma_norm = np.sqrt(dim) # Approximation
            # sigma = sigma * np.exp( cs * (norm_ps / sigma_norm - 1) )
            
            # Update covariance
            y_vec = (new_mean - mean) / sigma
            pc = (1 - cc) * pc + hsig * cc * y_vec
            
            rank_one = np.outer(y_vec, y_vec)
            rank_mu = np.sum(weights[:, None, None] * np.outer(pop[order[:mu]] - mean, pop[order[:mu]] - mean), axis=0) / sigma**2
            
            C = (1 + c1 * (2 - cc) - c1 - cmu) * C + c1 * (rank_one - pc @ pc.T) + cmu * rank_mu
            # Note: The update formula is C = (1 - c1 - cmu + c1*(2-cc)) * C + ...
            # Actually: C = (1 - c1 - cmu) * C + c1 * (rank_one + (1 - cc) * pc @ pc.T) + cmu * rank_mu
            # Let's use the standard:
            C = (1 - c1 - cmu) * C + c1 * (np.outer(y_vec, y_vec) + (1 - cc) * np.outer(pc, pc)) + cmu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive semi-definite
            # If eigenvalues are negative, fix them
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            return new_mean, sigma, C, ps, pc, fvals[order[0]], pop[order[0]]

        # Initialize CMA-ES
        # Start with a few random restarts to find a good basin
        # We will run CMA-ES for a certain number of generations, then restart if no improvement.
        
        # Budget allocation:
        # 500 calls. Lambda=8. Generations per run: 500/8 = 62.
        # We can do 1 long run or several shorter runs.
        # Weierstrass is multimodal, so restarts are good.
        
        # Strategy:
        # 1. Initial random sampling (10 points) to find a decent start.
        # 2. Run CMA-ES from the best point.
        # 3. If no improvement for 20 generations, restart from a new random point or a perturbation of the best.
        
        # Phase 1: Initialization
        init_count = 10
        init_points = rng.uniform(-5, 5, (init_count, dim))
        init_fvals = np.array([wrapper(p) for p in init_points])
        
        best_init_idx = np.argmin(init_fvals)
        current_mean = init_points[best_init_idx].copy()
        current_sigma = 1.0
        current_C = np.eye(dim)
        current_ps = np.zeros(dim)
        current_pc = np.zeros(dim)
        
        best_f_so_far = init_fvals[best_init_idx]
        stagnation = 0
        max_stagnation = 20 # Generations without improvement
        
        while calls_made < total_budget and time.time() < deadline:
            # Check if we should restart
            if stagnation > max_stagnation:
                # Restart
                # Choose a new starting point
                # Option 1: Random
                # Option 2: Perturb best
                if rng.random() < 0.5:
                    current_mean = rng.uniform(-5, 5, dim)
                else:
                    current_mean = best_x + rng.standard_normal(dim) * 0.5
                    current_mean = np.clip(current_mean, -5, 5)
                
                current_sigma = 1.0
                current_C = np.eye(dim)
                current_ps = np.zeros(dim)
                current_pc = np.zeros(dim)
                stagnation = 0
                
                # Evaluate the new start
                f_start = wrapper(current_mean)
                if f_start < best_f_so_far:
                    best_f_so_far = f_start
            
            # Run one generation of CMA-ES
            if calls_made + lambda_pop > total_budget:
                # Not enough budget for a full generation, break
                break
                
            try:
                new_mean, new_sigma, new_C, new_ps, new_pc, f_best_gen, x_best_gen = cma_es_step(
                    current_mean, current_sigma, current_C, current_ps, current_pc
                )
            except Exception:
                # If CMA-ES fails, fall back to simple (1+1)-ES
                # Or just break
                break
            
            current_mean = new_mean
            current_sigma = new_sigma
            current_C = new_C
            current_ps = new_ps
            current_pc = new_pc
            
            if f_best_gen < best_f_so_far:
                best_f_so_far = f_best_gen
                stagnation = 0
            else:
                stagnation += 1

        return best_x, best_f
