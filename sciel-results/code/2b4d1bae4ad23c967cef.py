import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        # Wrapper to track calls and enforce deadline
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = best_f
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Dense initial scan to find promising basins
        # f22 (Gallagher 21 peaks) has many local minima. A dense scan helps
        # identify which basins contain the global minimum.
        # Use a grid + random points for initial exploration.
        
        # Grid scan: 3x3x3x3x3 = 243 points, but that's too many for 500 budget.
        # Let's use a coarser grid: 2x2x2x2x2 = 32 points at {-3, 3}
        # Plus some random points.
        
        # Actually, let's use a stratified sampling approach.
        # Split each dimension into 4 bins, sample one point per bin combination.
        # 4^5 = 1024 is too many. Let's use 3^5 = 243 points at {-3, 0, 3}.
        # That's 243 calls, leaving 257 for refinement.
        
        # Let's do a mix: 125 points at a 5x5x5x5x5 grid subset + random
        # Actually, let's be more strategic. Use a space-filling approach.
        
        # Generate candidate starting points
        candidates = []
        
        # Origin and structured points
        candidates.append(np.zeros(dim))
        candidates.append(np.ones(dim) * 2.0)
        candidates.append(np.ones(dim) * -2.0)
        candidates.append(np.ones(dim) * 1.0)
        candidates.append(np.ones(dim) * -1.0)
        
        # Grid points at {-3, -1, 1, 3}
        grid_vals = [-3.0, -1.0, 1.0, 3.0]
        for a in grid_vals:
            for b in grid_vals:
                for c in grid_vals:
                    for d in grid_vals:
                        for e in grid_vals:
                            candidates.append(np.array([a, b, c, d, e]))
        
        # Add random points
        n_random = 50
        candidates.extend(rng.uniform(-5, 5, (n_random, dim)))
        
        candidates = np.array(candidates)
        
        # Evaluate all candidates
        for x in candidates:
            if calls >= budget - 50 or time.time() > deadline - 10:
                break
            wrapped_func(x)
        
        # Keep the top 10 best points from the scan
        # We need to track which points were evaluated and their values
        # Since wrapped_func doesn't return history, let's re-evaluate the top candidates
        # Actually, we should track the evaluations properly.
        
        # Let's restart with a cleaner approach:
        # Phase 1: Evaluate a set of diverse starting points
        # Phase 2: For the top K starting points, run local optimization
        # Phase 3: Polish the best result
        
        # Reset for a cleaner implementation
        # Actually, the above already made calls. Let's continue from here.
        
        # We don't have the individual values from the scan since wrapped_func 
        # only updates best. Let's do a different approach:
        # Evaluate points one by one and keep track of the best few.
        
        # For now, let's just use the best_x found so far and do local refinement.
        # But we lost the information about which other points were good.
        
        # Let's take a different structural approach:
        # Use a simple evolutionary strategy with population-based search.
        
        # Phase 2: (1+λ)-ES with multiple offspring
        pop_size = 10
        mu = 1  # Only the best survives
        lambda_ = 5  # Number of offspring per generation
        
        # Initialize population with the best point and some diversity
        pop = [best_x.copy() for _ in range(pop_size)]
        f_pop = [best_f] * pop_size
        
        # Add some diversity
        for i in range(1, pop_size):
            pop[i] = best_x + rng.normal(0, 1.0, dim)
            pop[i] = np.clip(pop[i], -5, 5)
            f_pop[i] = wrapped_func(pop[i])
            if f_pop[i] < best_f:
                best_f = f_pop[i]
                best_x = pop[i].copy()
        
        # Sort by fitness
        order = np.argsort(f_pop)
        pop = [pop[i].copy() for i in order]
        f_pop = [f_pop[i] for i in order]
        
        # ES loop
        sigma = 1.0
        max_generations = (budget - calls - 10) // lambda_
        
        for gen in range(max_generations):
            if calls >= budget - 10 or time.time() > deadline - 5:
                break
            
            # Generate offspring from the best individual
            offspring = []
            f_offspring = []
            
            for k in range(lambda_):
                if calls >= budget - 10 or time.time() > deadline - 5:
                    break
                child = pop[0] + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                f_child = wrapped_func(child)
                offspring.append(child)
                f_offspring.append(f_child)
            
            if len(offspring) == 0:
                break
            
            # Select the best offspring
            best_idx = np.argmin(f_offspring)
            if f_offspring[best_idx] < f_pop[0]:
                pop[0] = offspring[best_idx].copy()
                f_pop[0] = f_offspring[best_idx]
                best_x = pop[0].copy()
                best_f = f_pop[0]
            
            # Adapt sigma
            # Simple rule: if we found a better solution, increase sigma slightly
            # Otherwise, decrease it
            if f_offspring[best_idx] < f_pop[0] + 1e-10:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Occasionally restart from a random point to escape local minima
            if gen % 20 == 19 and calls < budget - 50:
                x_rand = rng.uniform(-5, 5, dim)
                f_rand = wrapped_func(x_rand)
                if f_rand < f_pop[0]:
                    pop[0] = x_rand.copy()
                    f_pop[0] = f_rand
                    best_x = pop[0].copy()
                    best_f = f_rand
        
        # Phase 3: Final polish with a finer search around the best point
        if calls < budget - 10 and time.time() < deadline - 5:
            sigma_fine = 0.1
            for _ in range(20):
                if calls >= budget - 5 or time.time() > deadline - 2:
                    break
                x_trial = best_x + sigma_fine * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = wrapped_func(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
        
        return best_x, best_f
