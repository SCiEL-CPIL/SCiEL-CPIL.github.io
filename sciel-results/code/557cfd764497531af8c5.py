import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse Exploration (LHS + Random)
        # Rastrigin is highly multimodal. We need to cover the space well.
        explore_budget = max(20, int(budget * 0.2))
        
        # Latin Hypercube Sampling for structured coverage
        n_lhs = min(explore_budget, 25)
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + d * 10.0 / dim
                hi = lo + 10.0 / dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)
            
        # Fill remaining exploration with random points
        remaining_explore = explore_budget - n_lhs
        for _ in range(remaining_explore):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        if best_x is None:
            x0 = np.zeros(dim)
            eval_point(x0)

        # Phase 2: Differential Evolution (DE)
        # DE is robust for multimodal problems and maintains population diversity.
        # We use a moderate population size to balance exploration and exploitation.
        remaining_budget = budget - call_count
        if remaining_budget < 50:
            return best_x, best_f

        pop_size = 20  # Fixed reasonable size for dim=5
        if remaining_budget < pop_size * 5:
            pop_size = max(10, remaining_budget // 5)
            
        # Initialize population from best found and random points
        pop = np.empty((pop_size, dim))
        pop_f = np.empty(pop_size)
        
        # Seed with best
        pop[0] = best_x
        pop_f[0] = best_f
        
        for i in range(1, pop_size):
            if call_count >= budget or time.time() > deadline:
                break
            # Mix of points near best and random points to maintain diversity
            if i % 2 == 0:
                x = best_x + rng.normal(0, 1.5, dim)
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            x = np.clip(x, -5.0, 5.0)
            pop[i] = x
            pop_f[i] = eval_point(x)
            
        # If we broke early due to budget, adjust pop_size
        actual_pop_size = min(pop_size, call_count - explore_budget) 
        # Note: The above logic is slightly tricky because eval_point increments call_count.
        # Let's just track indices properly.
        
        # Re-initialize properly to ensure consistency
        # We already have pop[0] and potentially pop[1..k] evaluated.
        # Let's restart the population setup cleanly if we didn't fill it.
        # Actually, the loop above filled as much as it could.
        # Let's just use the filled portion.
        
        # Determine actual number of valid population members
        # pop_f contains values for indices 0 to (pop_size-1) potentially.
        # But if call_count hit budget, some might be uninitialized.
        # Let's be safe:
        # --- Safe Population Initialization ---
        # Reset population
        pop = np.empty((pop_size, dim))
        pop_f = np.empty(pop_size)
        
        # 0: Best
        pop[0] = best_x
        pop_f[0] = best_f
        
        filled = 1
        while filled < pop_size:
            if call_count >= budget or time.time() > deadline:
                break
            if filled % 2 == 0:
                x = best_x + rng.normal(0, 1.5, dim)
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            x = np.clip(x, -5.0, 5.0)
            pop[filled] = x
            pop_f[filled] = eval_point(x)
            filled += 1
            
        actual_pop_size = filled
        if actual_pop_size < 5:
            return best_x, best_f

        # DE Parameters
        F = 0.8
        CR = 0.9
        
        # DE Loop
        while call_count < budget and time.time() < deadline:
            for i in range(actual_pop_size):
                if call_count >= budget or time.time() > deadline:
                    break
                    
                # Mutation
                # Select r1, r2, r3 distinct from i
                candidates = [j for j in range(actual_pop_size) if j != i]
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # DE/rand/1
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                        
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                trial_f = eval_point(trial)
                
                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    # Update global best if needed (eval_point already did this, 
                    # but we need to keep pop[best_idx] consistent if we track it)
                    # Since eval_point updates best_x/best_f globally, we just need 
                    # to ensure pop reflects the current state.
                    # We don't explicitly track best_idx in pop array anymore, 
                    # we just use best_x from the nonlocal scope.
                    # However, DE/best/1 would require knowing which index is best.
                    # We are using DE/rand/1 which doesn't need best_idx.

        # Phase 3: Local Refinement (1+1-ES)
        # Use the best point found by DE for fine-tuning.
        if call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            local_budget = min(budget - call_count, 100)
            
            for _ in range(local_budget):
                if call_count >= budget or time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        break

        return best_x, best_f
