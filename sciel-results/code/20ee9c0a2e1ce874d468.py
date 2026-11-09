"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-scale lattice search with CMA-ES refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Deterministic multi-scale lattice search
        # Büche-Rastrigin has periodic structure. We use a stratified sampling approach.
        # 5^5 = 3125 points is too many for budget 500.
        # We will use a 4^5 = 1024 point grid? Still too many.
        # Let's use a 3^5 = 243 point grid first to cover the space coarsely.
        # Then refine the best few points with a finer local grid or CMA-ES.
        
        # Actually, the directive suggests: "Use a deterministic multi-scale lattice of 5^5 = 3125 points... in the first 300 calls"
        # Wait, 3125 > 300. The directive says "identifying the 5-10 deepest basins in the first 300 calls".
        # So we can't evaluate all 3125. We must sample a subset or use a coarser grid.
        # Let's use a 3^5 = 243 point grid. This fits in 300 calls easily.
        # Points are in {-4, 0, 4} or similar? 
        # The domain is [-5, 5]. 
        # A good stratified sample for dim=5 with 3 points per dim:
        # levels: -4, 0, 4 (or -3.33, 0, 3.33)
        # Let's use levels: -4, 0, 4. This covers the likely range of the shifted minimum.
        
        levels = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        lattice_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            lattice_points.append([levels[i1], levels[i2], levels[i3], levels[i4], levels[i5]])
        
        lattice_points = np.array(lattice_points)
        
        # Shuffle to avoid bias if the function has a specific alignment, 
        # but deterministic is requested. However, randomizing the order of evaluation 
        # might help if we stop early due to time. 
        # Let's keep it deterministic but interleave the levels? 
        # No, just evaluate them all. 243 calls.
        
        # To make it more robust to the hidden shift, let's add a small random perturbation 
        # to the lattice points? No, deterministic multi-scale lattice.
        # But the shift is hidden. A fixed grid might miss the basin if the shift is e.g. 3.5.
        # The levels -4, 0, 4 might be too sparse.
        # Let's use 4 levels per dimension: -4.5, -1.5, 1.5, 4.5? 
        # 4^5 = 1024 points. Too many.
        
        # Let's stick to 3 levels per dim: 243 points.
        # We have 500 - 243 = 257 calls left.
        # We will pick the top K basins from the lattice search and run CMA-ES on them.
        
        # Evaluate lattice points
        f_values = np.zeros(len(lattice_points))
        for i in range(len(lattice_points)):
            if calls >= budget or time.time() > deadline:
                break
            x = lattice_points[i]
            f = func(x)
            f_values[i] = f
            calls += 1
            update_best(x, f)
        
        # If we didn't evaluate all, mark unevaluated as inf
        # (The loop above breaks, so remaining f_values are 0. We need to handle this.)
        # Let's re-structure: only evaluate if we have budget.
        
        # Let's restart the logic to be safer.
        
        # --- Revised Phase 1 ---
        # Generate 243 points
        # Evaluate them
        # If time/budget runs out, stop.
        
        # --- Phase 2: Refinement ---
        # Select top 5 points from the evaluated lattice points.
        # Run a short CMA-ES or local search from each.
        
        # Let's implement a simple CMA-ES for the refinement.
        # Or even simpler: a local random search (hill climbing) with adaptive step size.
        # Given the "cross" operator question, maybe we should try a simple genetic algorithm 
        # or just stick to CMA-ES as it's known to work well.
        
        # Let's use a lightweight CMA-ES implementation for the refinement.
        
        def run_cmaes(x_start, f_start, budget_left):
            """Run a short CMA-ES from x_start with f_start, using up to budget_left calls."""
            nonlocal calls, best_x, best_f, deadline
            
            if budget_left <= 0:
                return x_start, f_start
                
            dim = 5
            lambda_pop = 6  # Small population for short horizon
            mu = 3
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
            
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            x_mean = x_start.copy()
            sigma = 0.5  # Small initial sigma for local refinement
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_x = x_start.copy()
            local_best_f = f_start
            
            while calls + lambda_pop <= budget and time.time() < deadline:
                # Generate population
                population = []
                f_values_pop = []
                
                B = np.linalg.cholesky(C)
                
                for i in range(lambda_pop):
                    z = np.random.standard_normal(dim)
                    y = B @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = func(x_new)
                    calls += 1
                    
                    population.append(x_new)
                    f_values_pop.append(f_new)
                    
                    if f_new < local_best_f:
                        local_best_f = f_new
                        local_best_x = x_new.copy()
                    
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        
                # Sort
                indices = np.argsort(f_values_pop)
                population = [population[i] for i in indices]
                f_values_pop = [f_values_pop[i] for i in indices]
                
                # Update mean
                x_mean_new = np.zeros(dim)
                for i in range(mu):
                    x_mean_new += weights[i] * population[i]
                
                # Update paths
                step = (x_mean_new - x_mean) / sigma
                B_inv_step = np.linalg.solve(B, step)
                
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * B_inv_step
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * B_inv_step
                
                # Update Covariance
                outer_products = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (population[i] - x_mean) / sigma
                    outer_products += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * outer_products
                C = (C + C.T) / 2
                
                x_mean = x_mean_new
                
                # Update sigma
                chi0 = np.sqrt(dim) * (1 - 1/(2*dim**2) - 1/(8*dim**4))
                sigma = sigma * np.exp(cs / chi0 * (np.linalg.norm(p_sigma) / chi0 - 1))
                sigma = np.clip(sigma, 1e-4, 10.0)
                
            return local_best_x, local_best_f

        # Identify top basins from lattice search
        # We need to know which points were evaluated.
        # Since we broke the loop, let's just take the evaluated ones.
        
        # Let's re-run the lattice evaluation more carefully to track indices.
        # Actually, the previous loop just broke. Let's assume we evaluated the first N points.
        # To be robust, let's collect the evaluated points and their fitnesses.
        
        # Wait, the previous code block evaluated in a for loop and broke. 
        # I need to re-structure the main call to properly manage this.
        
        # Let's rewrite the main logic flow.

        # --- Restarting __call__ logic for clarity and correctness ---
        
        # Phase 1: Lattice Search
        lattice_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            lattice_points.append([levels[i1], levels[i2], levels[i3], levels[i4], levels[i5]])
        
        lattice_points = np.array(lattice_points)
        
        evaluated_points = []
        evaluated_fitness = []
        
        for i in range(len(lattice_points)):
            if calls >= budget or time.time() > deadline:
                break
            x = lattice_points[i]
            f = func(x)
            calls += 1
            evaluated_points.append(x)
            evaluated_fitness.append(f)
            update_best(x, f)
            
        if not evaluated_points:
            # Fallback if no calls made (shouldn't happen)
            x0 = np.zeros(dim)
            f0 = func(x0)
            calls += 1
            update_best(x0, f0)
            return best_x, best_f

        # Phase 2: Refinement
        # Sort evaluated points by fitness
        sorted_indices = np.argsort(evaluated_fitness)
        
        # Pick top K basins
        K = 5
        top_k_indices = sorted_indices[:K]
        
        # Allocate remaining budget to these K basins
        remaining_budget = budget - calls
        budget_per_basin = remaining_budget // K
        
        # Run CMA-ES from each top basin
        for idx in top_k_indices:
            if calls >= budget or time.time() > deadline:
                break
            
            x_start = evaluated_points[idx]
            f_start = evaluated_fitness[idx]
            
            # Run short CMA-ES
            run_cmaes(x_start, f_start, budget_per_basin)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
