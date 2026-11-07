"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE/rand/1/bin) optimizer with adaptive mutation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = max(10, 10 * dim)
        
        # Initialize population uniformly in [-5, 5]
        # Use a fixed seed for reproducibility in evaluation if needed, 
        # but random is fine as long as it's consistent per instance.
        # The evaluator creates a new instance for each test case.
        rng = np.random.default_rng(42)
        
        population = rng.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fvals = np.empty(pop_size)
        calls_made = 0
        for i in range(pop_size):
            if calls_made >= budget:
                break
            fvals[i] = func(population[i])
            calls_made += 1
            
        # Track best
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        # DE Parameters
        # F: mutation scale factor, CR: crossover rate
        # Standard DE uses F in [0.1, 1.0], CR in [0.1, 0.9]
        # We can adapt F based on progress or keep it constant.
        # For robustness on BBOB, F=0.5 and CR=0.5 are good defaults.
        # Let's use a slightly higher CR for better exploration in high dimensions.
        F = 0.5
        CR = 0.5
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate trial vector for each individual in population
            # We process one individual at a time to allow early termination if budget is tight
            # But DE requires a full generation to update the population properly.
            # Given budget=500 and pop_size=50, we can do ~8-10 generations.
            
            # To maximize efficiency, we should process the whole population.
            # If we run out of budget mid-generation, we just stop.
            
            # Shuffle indices for diversity in DE/rand/1
            perm = rng.permutation(pop_size)
            
            # We will update the population in-place or use a new one?
            # Standard DE: compare trial with current individual.
            # We need to be careful with call counting.
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals r1, r2, r3 different from i
                # Simple way: pick 3 random indices, ensure they are not i and distinct
                # For pop_size >= 4, this is easy.
                
                # Get 3 distinct random indices
                r_indices = rng.choice(pop_size, 3, replace=False)
                # Ensure none is i
                # If any is i, replace it
                # Since pop_size is small, we can just loop
                while any(r == i for r in r_indices):
                    r_indices = rng.choice(pop_size, 3, replace=False)
                    
                r1, r2, r3 = r_indices
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: uniform crossover
                # Jrand: random integer in [0, dim-1] to ensure at least one gene is from v
                Jrand = rng.integers(0, dim)
                
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == Jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fvals[i]:
                    population[i] = trial
                    fvals[i] = f_trial
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
