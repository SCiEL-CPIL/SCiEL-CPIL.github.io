"""Multi-Island CMA-ES with Error Repair for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Island CMA-ES with error repair for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        
        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        
        # Strategy: Multi-Island CMA-ES
        # We split the budget among 4 islands.
        # Each island runs its own CMA-ES.
        # Periodically, we perform "error repair":
        # If an island is stagnating, we inject the best solution from the best-performing island,
        # but we "repair" it by adding noise to escape the local minimum, effectively creating a new seed.
        
        n_islands = 4
        budget_per_island = total_budget // n_islands
        
        # CMA-ES parameters
        # For dim=5, lambda=8 is standard.
        lam = 8
        mu = max(1, lam // 2) # 4
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs
        cm = (2 + 2 / max(mu_eff, (lam - mu_eff) / 2)) / (dim**2 + 2)
        c_sigma = cs
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Initialize islands
        # To maximize diversity, we initialize them in different quadrants or random locations.
        # Lunacek has a secondary minimum, so diverse starts are crucial.
        islands = []
        
        # Initial random sampling to seed the islands
        # Use 20% of budget for initial exploration?
        # No, let's just start each island with a random mean.
        # To ensure coverage, we can use a simple space-filling approach for the first few points of each island.
        
        for i in range(n_islands):
            # Random start in [-5, 5]^dim
            # We can bias starts slightly to cover corners?
            # Let's just use uniform random for now, but ensure they are distinct.
            mean = rng.uniform(-5, 5, size=dim)
            
            island_state = {
                'mean': mean,
                'sigma': 2.0, # Initial step size
                'C': np.eye(dim),
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'best_x': mean,
                'best_f': np.inf,
                'calls_made': 0,
                'gen': 0,
                'stagnation_count': 0,
                'last_best_f': np.inf
            }
            islands.append(island_state)
        
        # Helper to run one generation of CMA-ES for a specific island
        def run_cma_generation(island, func, rng):
            if island['calls_made'] + lam > budget_per_island:
                return False
            
            mean = island['mean']
            sigma = island['sigma']
            C = island['C']
            
            # 1. Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim) + 1e-6 * np.eye(dim)
                island['C'] = C
                L = np.linalg.cholesky(C)
                
            z = rng.standard_normal((lam, dim))
            pop = mean + sigma * (z @ L.T)
            
            # 2. Evaluate
            fvals = np.array([func(x) for x in pop])
            island['calls_made'] += lam
            
            # 3. Update island best
            gen_best_idx = np.argmin(fvals)
            if fvals[gen_best_idx] < island['best_f']:
                island['best_f'] = fvals[gen_best_idx]
                island['best_x'] = pop[gen_best_idx].copy()
                island['stagnation_count'] = 0
            else:
                island['stagnation_count'] += 1
            
            # 4. Selection
            order = np.argsort(fvals)
            selected_indices = order[:mu]
            
            # 5. Update Mean
            new_mean = np.sum(weights[:, None] * pop[selected_indices], axis=0)
            
            # 6. Update Covariance and Step Size
            y = (pop[selected_indices] - mean) / sigma
            y = y.T # dim x mu
            
            # Update p_sigma
            island['p_sigma'] = (1 - c_sigma) * island['p_sigma'] + c_sigma * np.sqrt(mu_eff) * (weights @ y)
            
            # Update p_c
            island['p_c'] = (1 - c_c) * island['p_c'] + c_c * np.sqrt(mu_eff) * (weights @ y)
            
            # h_sigma check
            gen = island['gen']
            denom_sigma = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            denom_c = np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))
            
            if denom_sigma > 1e-10:
                threshold = (1.4 + 2 / (dim + 1)) * denom_c
                if np.linalg.norm(island['p_sigma']) / denom_sigma < threshold:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
            else:
                h_sigma = 1.0
            
            # Rank-one update
            C = (1 - cm * (1 + (1 - h_sigma) * c_sigma * c_sigma)) * C + cm * h_sigma * np.outer(island['p_sigma'], island['p_sigma'])
            
            # Rank-mu update
            Y_weighted = np.zeros((dim, dim))
            for i in range(mu):
                Y_weighted += weights[i] * np.outer(y[:, i], y[:, i])
            
            correction = 1 + (1 - h_sigma) * c_sigma * c_sigma
            C = (1 - cm * correction) * C + cm * correction * Y_weighted
            
            # Symmetrize
            C = (C + C.T) / 2
            island['C'] = C
            
            # Update sigma
            sigma *= np.exp((cs / ds) * (np.linalg.norm(island['p_sigma']) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            island['sigma'] = sigma
            
            # Update mean
            island['mean'] = new_mean
            island['gen'] += 1
            
            return True

        # Main Loop
        # We iterate until all islands are exhausted or time runs out.
        # In each iteration, we pick the island with the most remaining budget and run one generation.
        # This ensures balanced usage.
        
        while True:
            if time.time() - start_time > max_time:
                break
            
            # Find island with remaining budget
            active_islands = [i for i in range(n_islands) if islands[i]['calls_made'] < budget_per_island]
            if not active_islands:
                break
            
            # Pick island with max remaining budget to balance
            current_island_idx = max(active_islands, key=lambda i: budget_per_island - islands[i]['calls_made'])
            island = islands[current_island_idx]
            
            # Run one generation
            success = run_cma_generation(island, func, rng)
            if not success:
                continue
            
            # Update global best
            if island['best_f'] < global_best_f:
                global_best_f = island['best_f']
                global_best_x = island['best_x'].copy()
            
            # Error Repair / Migration Logic
            # Every 10 generations (or when stagnating), we might migrate.
            # "Error repair" concept: If an island is stuck (stagnation_count > threshold),
            # we "repair" its state by injecting the best solution from the best island,
            # but we perturb it to create a new exploration direction.
            
            if island['stagnation_count'] > 15:
                # Find the best island (excluding current if it's the best)
                best_island_idx = min(range(n_islands), key=lambda i: islands[i]['best_f'])
                
                if best_island_idx != current_island_idx:
                    best_island = islands[best_island_idx]
                    
                    # Repair: Reset current island's mean to best island's best solution
                    # but add noise to escape the local minimum.
                    # This is effectively a restart with a guided seed.
                    noise = rng.normal(0, 0.5, size=dim)
                    new_mean = best_island['best_x'] + noise
                    
                    # Reset CMA-ES state for this island
                    island['mean'] = new_mean
                    island['sigma'] = 1.0 # Moderate step size
                    island['C'] = np.eye(dim)
                    island['p_sigma'] = np.zeros(dim)
                    island['p_c'] = np.zeros(dim)
                    island['stagnation_count'] = 0
                    # Keep best_f and best_x? 
                    # If we reset, we might forget the local best. 
                    # But for CMA-ES, the "best" is used for selection. 
                    # If we restart the distribution, we should probably keep the historical best 
                    # for the global tracker, but the island's local best should reflect the new distribution's progress?
                    # Actually, if we keep the old best, and the new points are worse, stagnation will count up again.
                    # Let's reset the local best to the new mean's value? No, that's expensive (1 extra call).
                    # Let's just reset the stagnation counter and let the new generation update it.
                    # To avoid immediate stagnation trigger, we set last_best_f to the current global best?
                    # No, island best is local.
                    # Let's just reset the counter.
                    
                    # Optional: If the new mean is out of bounds, clip it?
                    # The function clips, so it's safe.
                    
        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
