"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # For dim=5, a population size of 10-15 is typical. 
        # We use 10 to balance exploration and budget usage.
        pop_size = 10
        
        # Initial best
        best_f = np.inf
        best_x = None
        
        # We will do a few restarts if we stall
        max_restarts = 5
        calls_per_restart = budget // max_restarts
        
        for restart in range(max_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Initialize population
            # Use a mix of random points and points near current best (if available)
            if best_x is None:
                pop = rng.uniform(-5, 5, (pop_size, dim))
            else:
                # 50% random, 50% perturbed best
                half = pop_size // 2
                pop_rand = rng.uniform(-5, 5, (half, dim))
                pop_best = np.tile(best_x, (pop_size - half, 1)) + rng.normal(0, 1.0, (pop_size - half, dim))
                pop = np.vstack([pop_rand, pop_best])
                pop = np.clip(pop, -5, 5)
            
            # Evaluate initial population
            fvals = np.array([func(x) for x in pop])
            calls_made = restart * (budget // max_restarts) # Rough estimate, we track below
            
            # Actually, let's track calls strictly.
            # The previous block was pseudocode for logic. Let's implement strictly.
            
            # Reset calls for this restart logic? No, global budget is strict.
            # Let's restructure to a single loop that manages the population state.
            
            pass # Placeholder to avoid confusion, logic below is the real one.

        # --- Real Implementation ---
        
        calls_made = 0
        best_f = np.inf
        best_x = None
        
        # We will use a simple (mu+lambda) DE-like strategy or just standard DE with restarts.
        # Given the small budget (500) and dim (5), we can afford a small population.
        # Let's use a population of 10.
        pop_size = 10
        
        # Initial Population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.zeros(pop_size)
        
        for i in range(pop_size):
            if time.time() > deadline or calls_made >= budget:
                break
            fvals[i] = func(pop[i])
            calls_made += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = pop[i].copy()
        
        # Main Loop
        # We will run DE until budget or time runs out.
        # If we stagnate, we restart a portion of the population.
        
        # Parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Stagnation tracking
        last_best = np.inf
        stagnation_counter = 0
        max_stagnation = 50 # Generations to wait before forcing diversity
        
        while calls_made < budget and time.time() < deadline:
            # Select best individual as reference (Best/1/DE)
            best_idx = np.argmin(fvals)
            base = pop[best_idx]
            
            new_pop = []
            new_fvals = []
            
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Don't use self as donor
                # Select 3 distinct random individuals other than i and best_idx
                donors = []
                while len(donors) < 3:
                    r = rng.integers(0, pop_size)
                    if r != i and r != best_idx and r not in donors:
                        donors.append(r)
                
                v = base + F * (pop[donors[0]] - pop[donors[1]])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                ft = func(trial)
                calls_made += 1
                
                # Selection
                if ft < fvals[i]:
                    new_pop.append(trial)
                    new_fvals.append(ft)
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                else:
                    new_pop.append(pop[i])
                    new_fvals.append(fvals[i])
            
            if not new_pop:
                break
                
            pop = np.array(new_pop)
            fvals = np.array(new_fvals)
            
            # Stagnation check
            if best_f < last_best - 1e-5:
                last_best = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # If stagnating, inject random individuals to escape local minima
            if stagnation_counter > max_stagnation:
                # Replace worst half with random
                worst_half = np.argsort(fvals)[-pop_size//2:]
                for idx in worst_half:
                    pop[idx] = rng.uniform(-5, 5, dim)
                    fvals[idx] = func(pop[idx])
                    calls_made += 1
                    if fvals[idx] < best_f:
                        best_f = fvals[idx]
                        best_x = pop[idx].copy()
                stagnation_counter = 0
                max_stagnation = max(20, max_stagnation // 2) # Reduce patience

        return best_x, best_f
# EVOLVE-BLOCK-END
