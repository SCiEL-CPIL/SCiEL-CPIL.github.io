"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Island Model CMA-ES for Multimodal Optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Island Model Parameters
        num_islands = 4
        # Each island gets a subset of the budget. 
        # We run them in parallel (conceptually), but sequentially in code, 
        # allocating budget dynamically based on remaining time and calls.
        # To ensure fair use, we divide the budget among islands for the main loop.
        
        calls_made = 0
        global_best_x = None
        global_best_f = float('inf')
        
        # Initialize islands with diverse starting points
        # Use a mix of center, corners, and random points to cover the box
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners (subset to avoid too many)
        corners = np.array(np.meshgrid([-4, 0, 4] * dim, indexing='ij')).T.reshape(-1, dim)
        # Sample a few corners
        corner_indices = rng.choice(len(corners), size=min(num_islands - 1, len(corners)), replace=False)
        for idx in corner_indices:
            start_points.append(corners[idx])
            
        # Pad with random points if needed
        while len(start_points) < num_islands:
            start_points.append(rng.uniform(-4, 4, dim))
            
        # Shuffle to randomize assignment
        rng.shuffle(start_points)
        
        # Allocate budget per island
        # Reserve 10% for final polishing of the global best
        polish_budget = int(budget * 0.1)
        main_budget = budget - polish_budget
        budget_per_island = main_budget // num_islands
        
        # CMA-ES Parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        weights = np.log(lambda_ / 2) - np.arange(1, lambda_ + 1)
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State for each island
        islands = []
        for i in range(num_islands):
            x0 = start_points[i].copy()
            sigma0 = 2.0
            C0 = np.eye(dim)
            pc0 = np.zeros(dim)
            ps0 = np.zeros(dim)
            islands.append({
                'x': x0,
                'sigma': sigma0,
                'C': C0,
                'pc': pc0,
                'ps': ps0,
                'best_x': x0,
                'best_f': float('inf'),
                'calls': 0
            })
        
        # Evaluate initial points for each island
        for island in islands:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = func(island['x'])
            calls_made += 1
            island['calls'] += 1
            island['best_f'] = fx
            island['best_x'] = island['x'].copy()
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = island['x'].copy()
        
        # Migration Interval: every N evaluations, share bests
        # With 4 islands and ~100 calls each, maybe migrate every 20 calls per island?
        # Simpler: Migrate after each island completes its allocated budget, or periodically.
        # Let's do periodic migration every 10 generations (each gen = lambda_ calls)
        # But since we run islands sequentially in this loop structure, we can just 
        # run one generation for each island in a round-robin fashion.
        
        # Round-Robin Execution
        # We will iterate until budget is exhausted or time runs out.
        # In each round, we try to run one generation for each island.
        
        round_num = 0
        while calls_made < budget and time.time() < deadline:
            round_num += 1
            # Check if all islands are done with their main budget
            all_done = True
            for island in islands:
                if island['calls'] < budget_per_island:
                    all_done = False
                    break
            
            if all_done:
                break
                
            # Run one generation for each active island
            for island in islands:
                if island['calls'] >= budget_per_island:
                    continue
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                    
                # CMA-ES Step
                try:
                    D = np.linalg.cholesky(island['C'])
                except np.linalg.LinAlgError:
                    island['C'] = np.eye(dim)
                    D = np.eye(dim)
                    
                pop = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = island['x'] + island['sigma'] * (D @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                    
                fvals = np.full(lambda_, np.inf)
                eval_count = 0
                for i in range(lambda_):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    fvals[i] = func(pop[i])
                    calls_made += 1
                    island['calls'] += 1
                    eval_count += 1
                    if fvals[i] < island['best_f']:
                        island['best_f'] = fvals[i]
                        island['best_x'] = pop[i].copy()
                    if fvals[i] < global_best_f:
                        global_best_f = fvals[i]
                        global_best_x = pop[i].copy()
                        
                if eval_count == 0:
                    continue
                    
                # Sort and update
                order = np.argsort(fvals)
                # Use only evaluated points for update if partial
                if eval_count < lambda_:
                    # This is tricky for CMA-ES. Standard CMA-ES assumes full population.
                    # If we cut short, we should probably just stop this island for this round
                    # or use a smaller lambda. For simplicity, if we can't complete a generation,
                    # we stop the island for this round.
                    continue

                x_mean = (pop[order[:int(mu_eff)]] * weights[:, None]).sum(axis=0)
                
                D_inv = np.linalg.inv(D)
                pc = (1 - c_c) * island['pc'] + c_c * (D_inv @ (x_mean - island['x']))
                norm_pc = np.linalg.norm(pc)
                if norm_pc < 1e-10:
                    norm_pc = 1e-10
                hsig = (1 - np.exp(-1 + c_c)) * chi_n
                ps = (1 - c_sigma) * island['ps'] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / norm_pc * (D_inv @ (x_mean - island['x']))
                
                delta_hsig = (1 - np.exp(-1 + c_c)) * (1 - np.exp(-2 * c_c))
                C_new = (1 + c_1 * delta_hsig - c_1 - c_mu) * island['C'] \
                    + c_1 * (np.outer(pc, pc) - island['C'])
                    
                diff = (pop[order[:int(mu_eff)]] - island['x']).T
                C_new += c_mu / mu_eff * (diff * (weights[:int(mu_eff)] ** 2)[None, :]) @ diff
                
                C_new = (C_new + C_new.T) / 2
                
                # Ensure PD
                eigvals, eigvecs = np.linalg.eigh(C_new)
                if np.any(eigvals < 1e-10):
                    C_new = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

                island['sigma'] *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                island['sigma'] = np.clip(island['sigma'], 1e-10, 10.0)
                
                island['x'] = x_mean
                island['C'] = C_new
                island['pc'] = pc
                island['ps'] = ps

            # Migration: After a few rounds, share the best solution
            # Let's migrate every 5 rounds (approx 5 * lambda * num_islands calls)
            if round_num % 5 == 0:
                # Find the current global best among islands
                best_island_idx = np.argmin([isl['best_f'] for isl in islands])
                best_island = islands[best_island_idx]
                
                # Migrate: Replace the worst island's state with a copy of the best, 
                # but with a slightly larger sigma to explore around it? 
                # Or just reset the worst island to the best location.
                worst_island_idx = np.argmax([isl['best_f'] for isl in islands])
                if worst_island_idx != best_island_idx:
                    worst_island = islands[worst_island_idx]
                    # Reset worst island to best location, but keep some diversity
                    worst_island['x'] = best_island['x'].copy() + rng.normal(0, 0.5, dim)
                    worst_island['x'] = np.clip(worst_island['x'], -5, 5)
                    worst_island['sigma'] = 1.0 # Moderate restart
                    worst_island['C'] = np.eye(dim)
                    worst_island['pc'] = np.zeros(dim)
                    worst_island['ps'] = np.zeros(dim)
                    # Note: We do NOT reset worst_island's best_f/x to global best immediately 
                    # because it hasn't evaluated there yet. But we can seed its search there.
        
        # Phase 2: Polishing
        remaining_budget = budget - calls_made
        if remaining_budget > 5 and global_best_x is not None and time.time() < deadline:
            x = global_best_x.copy()
            sigma = 0.1
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Small population CMA-ES
            lambda_p = 4
            weights_p = np.log(lambda_p / 2) - np.arange(1, lambda_p + 1)
            weights_p /= weights_p.sum()
            mu_eff_p = 1 / (weights_p ** 2).sum()
            c_sigma_p = (mu_eff_p + 2) / (dim + mu_eff_p + 5)
            c_c_p = (4 + mu_eff_p / dim) / (dim + 4 + 2 * mu_eff_p / dim)
            c_1_p = 2 / ((dim + 1.3) ** 2 + mu_eff_p)
            c_mu_p = min(1 - c_1_p, 2 * (mu_eff_p - 2 + 1 / mu_eff_p) / ((dim + 2) ** 2 + mu_eff_p))
            chi_n_p = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            while calls_made < budget and time.time() < deadline:
                try:
                    D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    D = np.eye(dim)
                    
                pop = np.zeros((lambda_p, dim))
                for i in range(lambda_p):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = x + sigma * (D @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                    
                fvals = np.full(lambda_p, np.inf)
                eval_done = 0
                for i in range(lambda_p):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    fvals[i] = func(pop[i])
                    calls_made += 1
                    eval_done += 1
                    if fvals[i] < global_best_f:
                        global_best_f = fvals[i]
                        global_best_x = pop[i].copy()
                        
                if eval_done < lambda_p:
                    break
                    
                order = np.argsort(fvals)
                x_mean = (pop[order[:int(mu_eff_p)]] * weights_p[:, None]).sum(axis=0)
                
                D_inv = np.linalg.inv(D)
                pc = (1 - c_c_p) * pc + c_c_p * (D_inv @ (x_mean - x))
                norm_pc = np.linalg.norm(pc)
                if norm_pc < 1e-10: norm_pc = 1e-10
                hsig = (1 - np.exp(-1 + c_c_p)) * chi_n_p
                ps = (1 - c_sigma_p) * ps + np.sqrt(c_sigma_p * (2 - c_sigma_p) * mu_eff_p) / norm_pc * (D_inv @ (x_mean - x))
                
                delta_hsig = (1 - np.exp(-1 + c_c_p)) * (1 - np.exp(-2 * c_c_p))
                C_new = (1 + c_1_p * delta_hsig - c_1_p - c_mu_p) * C \
                    + c_1_p * (np.outer(pc, pc) - C)
                diff = (pop[order[:int(mu_eff_p)]] - x).T
                C_new += c_mu_p / mu_eff_p * (diff * (weights_p[:int(mu_eff_p)] ** 2)[None, :]) @ diff
                C_new = (C_new + C_new.T) / 2
                
                eigvals, eigvecs = np.linalg.eigh(C_new)
                if np.any(eigvals < 1e-10):
                    C_new = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

                sigma *= np.exp(c_sigma_p / chi_n_p * (np.linalg.norm(ps) / chi_n_p - 1))
                sigma = np.clip(sigma, 1e-10, 1.0)
                
                x = x_mean
                C = C_new
                pc = pc
                ps = ps

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
