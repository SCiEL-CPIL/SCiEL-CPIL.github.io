import numpy as np
import time
import math

class Optimizer:
    """
    Island Model CMA-ES Optimizer.
    
    Strategy:
    Instead of a single global population or multi-start local searches, 
    this strategy runs K independent CMA-ES populations (islands) in parallel.
    Each island maintains its own mean, covariance matrix, and step size.
    Periodically, the best individual from each island is migrated to the 
    second-best island to inject diversity and prevent premature convergence 
    in suboptimal local minima.
    
    This structural change addresses the question of whether 'island' dynamics 
    improve the chance of finding the global minimum in a complex, 
    potentially multimodal landscape (Griewank-Rosenbrock with rotation/shift).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        # Configuration for Island Model
        # K islands
        K = 4 
        # Population size per island (lambda)
        # Total evals per generation = K * lambda
        # We need to fit within 500 evals.
        # Let's try lambda = 10 per island. Total 40 per generation.
        # 500 / 40 = 12.5 generations.
        # This is a bit low for CMA-ES to converge from scratch, but with islands, 
        # we might find good regions faster.
        # Alternatively, smaller islands: K=8, lambda=6 -> 48 per gen -> ~10 gens.
        # Let's stick to K=4, lambda=12 -> 48 per gen -> ~10 gens.
        # Actually, CMA-ES performs better with larger lambda.
        # Let's use K=3 islands, lambda=15. Total 45 per gen.
        # 500 / 45 = 11 generations.
        
        # Migration: Every M generations, migrate best 1 from each to worst 1?
        # Or just broadcast best to all?
        # Simple migration: Every 3 generations, replace the worst individual 
        # in each island with the best individual from the global set.
        
        n_islands = 4
        pop_size = 10  # Total evals per cycle = 40
        
        # Initialize Islands
        # Each island has: mean (mu), sigma (step size), C (covariance matrix), 
        #                   evolution paths p_sigma, p_c
        
        # Standard CMA-ES parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_s = (2 + mu_eff / (dim + 2)) / (dim + 2 + 2 * mu_eff / dim)
        
        # Initialize each island with different starting points
        rng = np.random.RandomState(123)
        
        islands = []
        for i in range(n_islands):
            # Spread initial means across the domain
            # Use a structured grid or random with large spread
            if i == 0:
                init_mean = np.zeros(dim)
            else:
                # Random initialization in a larger box to ensure diversity
                init_mean = rng.uniform(-4, 4, dim)
            
            # Initial sigma: large enough to explore, small enough to not overshoot?
            # Box is [-5, 5]. Range 10.
            # sigma0 = 2.0 is a reasonable start for 5D.
            sigma = 2.0
            
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Store population history for this island (not strictly needed if we evaluate on the fly, 
            # but we need the current generation's offspring)
            
            islands.append({
                'mean': init_mean,
                'sigma': sigma,
                'C': C,
                'p_sigma': p_sigma,
                'p_c': p_c,
                'best_x': init_mean,
                'best_fx': np.inf
            })
        
        # Pre-generate random directions for efficiency? No, CMA-ES needs normal sampling from C.
        # We will sample on the fly.
        
        def evaluate(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            f_val = func(x)
            calls += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = x.copy()
            return f_val
        
        generation = 0
        total_evals_per_cycle = n_islands * pop_size
        
        while calls < budget and time.time() < deadline:
            # Check if we can afford another full cycle
            if calls + total_evals_per_cycle > budget:
                # If we can't afford a full cycle, we might do partial evaluation
                # or just stop to be safe. 
                # Given the strict "more than 500 calls invalid" rule, we should stop 
                # if we can't guarantee we won't exceed.
                # However, we can evaluate one by one.
                # For simplicity and safety, we will break if the next full cycle exceeds budget.
                # This might waste some budget if budget % total_evals_per_cycle != 0,
                # but it's safer.
                # Actually, we can just evaluate until budget is hit.
                pass
            
            # 1. Generate offspring for each island
            all_offspring = []
            for k in range(n_islands):
                isl = islands[k]
                mean = isl['mean']
                sigma = isl['sigma']
                C = isl['C']
                
                # Compute Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite (shouldn't happen often), reset or use identity
                    L = np.eye(dim)
                
                # Sample normal vectors
                # z ~ N(0, I)
                z = rng.standard_normal((pop_size, dim))
                # Transform to x ~ N(mean, sigma^2 * C)
                # x = mean + sigma * z @ L.T
                offspring = mean + sigma * (z @ L.T)
                
                # Clip to bounds? The function clips, but for CMA-ES updates, 
                # we should probably use the unclipped values for the step calculation 
                # or handle bounds carefully. 
                # Standard CMA-ES doesn't handle bounds well. 
                # Since func clips, we can pass unclipped. 
                # However, if mean goes far out, sigma might need adjustment.
                # For now, we pass as is.
                
                for j in range(pop_size):
                    x = offspring[j]
                    # Evaluate
                    f_val = evaluate(x)
                    if calls >= budget or time.time() > deadline:
                        break
                    all_offspring.append((f_val, x, k, j))
                
                if calls >= budget or time.time() > deadline:
                    break
            
            if calls >= budget or time.time() > deadline:
                break
            
            # 2. Sort offspring within each island and update CMA-ES parameters
            # Group by island
            island_offspring = {k: [] for k in range(n_islands)}
            for f_val, x, k, j in all_offspring:
                island_offspring[k].append((f_val, x))
            
            for k in range(n_islands):
                if not island_offspring[k]:
                    continue
                
                # Sort by fitness
                island_offspring[k].sort(key=lambda t: t[0])
                
                # Select top mu
                selected = island_offspring[k][:mu]
                if len(selected) < mu:
                    continue
                
                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * selected[i][1]
                
                # Update best for island
                if selected[0][0] < islands[k]['best_fx']:
                    islands[k]['best_fx'] = selected[0][0]
                    islands[k]['best_x'] = selected[0][1].copy()
                
                # Compute step
                y = (new_mean - islands[k]['mean']) / islands[k]['sigma']
                
                # Update evolution paths
                p_sigma = islands[k]['p_sigma']
                p_c = islands[k]['p_c']
                C = islands[k]['C']
                
                # Need inverse square root of C for p_sigma update
                try:
                    C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)
                
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ y
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y
                
                # Update sigma
                sigma = islands[k]['sigma'] * math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
                
                # Update covariance
                delta_h = (1.1 + 0.5 * mu_eff / dim) * (np.linalg.norm(p_c)**2 / dim - 1)
                rank_one_update = np.outer(p_c, p_c)
                rank_mu_update = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (selected[i][1] - islands[k]['mean']) / islands[k]['sigma']
                    rank_mu_update += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_c * (2 - c_c) * delta_h - c_s) * C + c_s * (rank_one_update + rank_mu_update)
                
                # Enforce symmetry and positive definiteness (simple check)
                C = (C + C.T) / 2
                
                islands[k]['mean'] = new_mean
                islands[k]['sigma'] = sigma
                islands[k]['C'] = C
                islands[k]['p_sigma'] = p_sigma
                islands[k]['p_c'] = p_c
            
            generation += 1
            
            # 3. Migration
            # Every 3 generations, migrate the best individual from each island 
            # to the island with the second-best mean? 
            # Or simpler: Broadcast the global best to all islands' mean?
            # No, that collapses islands.
            # Standard migration: Replace the worst individual in the population 
            # of each island with the best individual from a random other island.
            # Since we don't keep the full population, we can just update the mean 
            # slightly towards the global best if the island is stagnating?
            
            # Let's do a simple migration: 
            # Find the global best individual found so far (best_x, best_fx).
            # For each island, if its best_fx is worse than the global best, 
            # and the island has been stagnant for a while, pull its mean towards the global best.
            
            # Simpler structural migration:
            # Every 4 generations, swap the 'mean' of the worst island with the 'mean' of the best island.
            # This preserves the covariance structure of the best island but moves it?
            # No, better: Copy the best individual (x) to be the new mean of the worst island?
            
            if generation % 4 == 0:
                # Find best and worst islands based on their current best_fx
                island_indices = list(range(n_islands))
                island_indices.sort(key=lambda k: islands[k]['best_fx'])
                
                best_island_idx = island_indices[0]
                worst_island_idx = island_indices[-1]
                
                # Migrate: Set the worst island's mean to the best island's best_x
                # This is a strong migration, effectively restarting the worst island 
                # near the best found solution but with its own covariance (which might be bad).
                # To be safer, we can also copy the covariance or reset it.
                # Let's just move the mean. The covariance will adapt.
                islands[worst_island_idx]['mean'] = islands[best_island_idx]['best_x'].copy()
                # Reset step size to a moderate value to allow exploration around the new mean
                islands[worst_island_idx]['sigma'] = 0.5 
                # Reset paths
                islands[worst_island_idx]['p_sigma'] = np.zeros(dim)
                islands[worst_island_idx]['p_c'] = np.zeros(dim)
                # Keep C or reset? Resetting to identity is safer if the old C was bad.
                islands[worst_island_idx]['C'] = np.eye(dim)

        return best_x, best_fx
