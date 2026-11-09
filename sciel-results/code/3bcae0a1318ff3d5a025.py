"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with global archive for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def check_time():
            return (time.time() - start_time) < time_limit

        # Parameters for DE
        # F is the differential weight, CR is the crossover rate
        F = 0.8
        CR = 0.9
        # Population size. For dim=5, 10*dim is standard, but we have 500 calls.
        # Let's use a moderate population size to allow for many generations.
        pop_size = 30
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        calls = pop_size
        
        # Global best
        idx_best = np.argmin(fitness)
        x_best = pop[idx_best].copy()
        f_best = fitness[idx_best]
        
        # Archive to maintain diversity and avoid premature convergence
        # We keep the best 'archive_size' points found so far, ensuring they are distinct enough
        archive_size = 20
        archive = [x_best.copy()]
        archive_f = [f_best]
        
        # Helper to add to archive
        def add_to_archive(x, f):
            nonlocal archive, archive_f
            if len(archive) < archive_size:
                archive.append(x.copy())
                archive_f.append(f)
            else:
                # If new point is better than the worst in archive, replace worst
                worst_idx = np.argmax(archive_f)
                if f < archive_f[worst_idx]:
                    # Check if it's sufficiently different from existing archive points
                    # Simple check: distance to nearest neighbor > 0.1
                    min_dist = np.min([np.linalg.norm(x - ax) for ax in archive])
                    if min_dist > 0.5:
                        archive[worst_idx] = x.copy()
                        archive_f[worst_idx] = f
                        # Re-sort archive for efficiency? Not strictly necessary for small size, 
                        # but helps if we want to pick random distinct points.
                        # For now, just replace.
        
        # Main DE loop
        while calls < budget and check_time():
            # Generate new population
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if not check_time() or calls >= budget:
                    break
                    
                # Select three distinct random indices different from i
                r1, r2, r3 = i
                while r1 == i:
                    r1 = rng.integers(0, pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, pop_size)
                
                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                x_trial = np.copy(pop[i])
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        x_trial[j] = v[j]
                
                # Clip to bounds
                x_trial = np.clip(x_trial, -5, 5)
                
                # Evaluate
                f_trial = func(x_trial)
                calls += 1
                new_pop[i] = x_trial
                new_fitness[i] = f_trial
                
                # Update global best
                if f_trial < f_best:
                    f_best = f_trial
                    x_best = x_trial.copy()
                    add_to_archive(x_best, f_best)
                    
                # Update archive if this point is good and distinct
                if f_trial < f_best + 1.0: # Only consider reasonably good points
                    add_to_archive(x_trial, f_trial)

            # Update population: keep the better of parent and child
            # This is (1+1) selection per individual, but since we generate a full new pop,
            # we can just replace the whole population if on average it's better, 
            # or do individual competition. Individual competition is standard DE.
            for i in range(pop_size):
                if new_fitness[i] < fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]
            
            # Occasionally inject random points to escape local minima if stagnation is detected
            # Simple stagnation check: if best hasn't improved in last 50 calls, inject randoms
            # We don't track history perfectly here, but we can check if f_best is stuck.
            # For simplicity and robustness, we rely on the archive and DE's exploration.
            # However, if we suspect we are stuck, we can replace a fraction of the population with random points.
            
            # Check for stagnation: if the best fitness hasn't changed in a while?
            # We can't easily track "last improvement" without a counter.
            # Let's add a counter.
            
            # Actually, let's just run until budget or time.
            
            # To ensure we don't get stuck in a single basin, we can occasionally
            # replace the worst individual in the population with a random point
            # or a point from the archive.
            if calls < budget:
                # Replace the worst individual in the current population with a random one
                # This helps with multimodality.
                worst_idx = np.argmax(fitness)
                # With some probability, replace with random
                if rng.random() < 0.1:
                    pop[worst_idx] = rng.uniform(-5, 5, dim)
                    fitness[worst_idx] = func(pop[worst_idx])
                    calls += 1
                    if fitness[worst_idx] < f_best:
                        f_best = fitness[worst_idx]
                        x_best = pop[worst_idx].copy()
                        add_to_archive(x_best, f_best)
                    add_to_archive(pop[worst_idx], fitness[worst_idx])

        return x_best, f_best
# EVOLVE-BLOCK-END
