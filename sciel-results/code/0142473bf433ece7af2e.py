"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Island Model CMA-ES for robust global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 20s safety margin
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # Parameters for CMA-ES
        # For dim=5, a small population is sufficient for each island
        # Total budget 500. If we have 4 islands, each gets ~125 evaluations.
        # Lambda=10 per island -> 12.5 generations per island.
        # This is very few generations. 
        # Maybe fewer islands? 2 islands?
        # Or larger lambda?
        # Let's try 3 islands.
        # 500 / 3 = 166 evals per island.
        # Lambda=10 -> 16 generations. Still few.
        # Lambda=20 -> 8 generations.
        
        # Actually, for 5D, CMA-ES is very fast.
        # Let's try 4 islands with Lambda=10.
        # Migration every 5 generations.
        
        n_islands = 4
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_lambda = lambda_ ** 0.5 * (1 - 1 / (4 * lambda_) - 1 / (25 * lambda_ ** 2))
        
        # Initialize Islands
        # Spread initial means across the domain to ensure diversity
        # Instead of random, use a grid or space-filling approach for initial means
        # For simplicity, random but with different sigmas?
        # Or just random.
        
        islands = []
        for i in range(n_islands):
            # Random start
            x_m = np.random.uniform(-5, 5, dim)
            # Start with a moderate sigma. 
            # Griewank has a global minimum at 0, but local minima exist.
            # A larger sigma helps escape local minima.
            sigma = 2.0 
            C = np.eye(dim)
            
            # Precompute C^{1/2} and C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            C_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_half = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Initial evaluation
            f_m = func(x_m)
            
            # Track best for this island
            best_x = x_m.copy()
            best_f = f_m
            
            islands.append({
                'x_m': x_m,
                'sigma': sigma,
                'C': C,
                'C_half': C_half,
                'C_inv_half': C_inv_half,
                'pc': pc,
                'ps': ps,
                'best_x': best_x,
                'best_f': best_f,
                'gen': 0
            })
            
            # Record global best
            global_best_x = best_x
            global_best_f = best_f
            
        calls_made = n_islands # Initial evals
        
        # Main Loop
        # We will run generations. Each generation consumes n_islands * lambda_ evaluations.
        # We stop when calls_made >= budget or time > deadline.
        
        # Migration counter
        migr_counter = 0
        migr_interval = 5 # Every 5 generations
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Check if we can afford a full generation
            gen_cost = n_islands * lambda_
            if calls_made + gen_cost > budget:
                # If we can't afford a full gen, we might do partial or stop.
                # For simplicity, stop if we can't do at least one island's worth?
                # Or just break.
                # Actually, we can do one island at a time.
                # But CMA-ES needs a full population to update.
                # So we must stop if we can't complete a generation.
                break
            
            # Evaluate all islands
            # We'll collect data for each island
            island_data = []
            
            for i in range(n_islands):
                isl = islands[i]
                x_m = isl['x_m']
                sigma = isl['sigma']
                C_half = isl['C_half']
                
                # Generate population
                Z = np.random.standard_normal((lambda_, dim))
                ys = x_m + sigma * (Z @ C_half.T)
                
                fs = np.zeros(lambda_)
                # Evaluate
                for j in range(lambda_):
                    if calls_made >= budget:
                        break
                    y_j = np.clip(ys[j], -5, 5)
                    f_j = func(y_j)
                    fs[j] = f_j
                    calls_made += 1
                    
                    # Update island best
                    if f_j < isl['best_f']:
                        isl['best_f'] = f_j
                        isl['best_x'] = y_j.copy()
                        
                    # Update global best
                    if f_j < global_best_f:
                        global_best_f = f_j
                        global_best_x = y_j.copy()
                        
                if calls_made >= budget:
                    break
                
                # Sort
                idx = np.argsort(fs)
                ys = ys[idx]
                fs = fs[idx]
                
                # Update CMA-ES parameters for this island
                w = weights
                x_m_new = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
                
                # Update ps
                delta_m = (x_m_new - x_m) / sigma
                C_inv_half = isl['C_inv_half']
                ps = (1 - c_sigma) * isl['ps'] + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_half @ delta_m)
                
                # Update sigma
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (isl['gen'] + 1)))) < (1.4 + 2 / (dim + 1)) * chi_lambda
                sigma_new = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / chi_lambda - 1))
                
                # Update pc
                if h_sigma:
                    pc = (1 - c_c) * isl['pc'] + c_c * delta_m
                else:
                    pc = (1 - c_c) * isl['pc']
                
                # Update C
                steps = (ys[:mu] - x_m) / sigma
                rank_mu_term = (steps.T * w) @ steps
                decay = 1 - c_1 - c_mu
                C_new = decay * isl['C'] + c_1 * np.outer(pc, pc) + c_mu * rank_mu_term
                C_new = (C_new + C_new.T) / 2
                
                # Eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C_new)
                eigvals = np.maximum(eigvals, 1e-20)
                C_half_new = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_half_new = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update island state
                isl['x_m'] = x_m_new
                isl['sigma'] = np.clip(sigma_new, 1e-12, 1e12)
                isl['C'] = C_new
                isl['C_half'] = C_half_new
                isl['C_inv_half'] = C_inv_half_new
                isl['pc'] = pc
                isl['ps'] = ps
                isl['gen'] += 1
                
                island_data.append(isl)
                
            if calls_made >= budget:
                break
            
            # Migration
            migr_counter += 1
            if migr_counter >= migr_interval:
                migr_counter = 0
                # Simple migration: Replace the worst island's mean with the best island's mean
                # Or exchange a few individuals?
                # Let's do "Best replaces Worst Mean" to inject diversity from successful areas.
                # Find best and worst island based on best_f
                best_idx = np.argmin([isl['best_f'] for isl in islands])
                worst_idx = np.argmax([isl['best_f'] for isl in islands])
                
                if best_idx != worst_idx:
                    # Copy mean and covariance from best to worst?
                    # This might be too aggressive.
                    # Instead, let's just reset the worst island's mean to the best island's mean,
                    # but keep its sigma and C? Or reset C too?
                    # Let's try: Worst island adopts the best island's x_m.
                    # This effectively creates a new branch from a good point.
                    islands[worst_idx]['x_m'] = islands[best_idx]['x_m'].copy()
                    # Maybe perturb it slightly to avoid exact overlap
                    islands[worst_idx]['x_m'] += 0.1 * np.random.standard_normal(dim)
                    # Reset its local history to avoid bias from the old trajectory?
                    # Let's reset ps and pc
                    islands[worst_idx]['ps'] = np.zeros(dim)
                    islands[worst_idx]['pc'] = np.zeros(dim)
                    # Keep sigma and C to maintain shape adaptation?
                    # Or reset C to Identity?
                    # Let's keep C to see if the shape is still useful.
                    
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
