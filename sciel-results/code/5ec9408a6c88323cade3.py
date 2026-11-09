"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        # DE Parameters
        # Population size typically 5*dim to 10*dim
        pop_size = 25 
        F = 0.7  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(lower, upper, (pop_size, n))
        fit = np.empty(pop_size)
        
        calls_made = 0
        for i in range(pop_size):
            fit[i] = func(pop[i])
            calls_made += 1
            
        best_idx = np.argmin(fit)
        best_val = fit[best_idx]
        best_pos = pop[best_idx].copy()
        
        if calls_made >= budget:
            return best_pos, best_val
            
        # Main loop
        # We have budget - calls_made calls left.
        # Each generation uses pop_size calls.
        # We can do floor(remaining / pop_size) generations.
        
        while calls_made < budget:
            # Check if we can do a full generation
            if budget - calls_made < pop_size:
                # Do partial generation or stop? 
                # DE usually works in full generations. 
                # If we have less than pop_size, we can evaluate a few random perturbations of the best.
                # Or just break.
                # Let's try to use remaining budget for local search around best or random restarts.
                # For simplicity, if we can't do a full gen, we stop. 
                # But we can squeeze out a few calls.
                # Let's just break if we can't do a full gen to keep logic simple, 
                # or we can evaluate random points.
                # Let's evaluate random points from the best if remaining > 0.
                while calls_made < budget:
                    # Local perturbation
                    step = 0.1 * (upper - lower)
                    trial = best_pos + rng.normal(0, step, n)
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    calls_made += 1
                    if f_trial < best_val:
                        best_val = f_trial
                        best_pos = trial
                break
                
            # Full Generation
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                    
                # Select 3 distinct random individuals different from i
                # Simple selection: random unique indices
                # To ensure distinctness and != i
                indices = [i]
                while len(indices) < 4:
                    idx = rng.integers(0, pop_size)
                    if idx not in indices:
                        indices.append(idx)
                
                i1, i2, i3 = indices[1], indices[2], indices[3]
                
                # Mutation
                v = pop[i1] + F * (pop[i2] - pop[i3])
                
                # Crossover
                j_rand = rng.integers(0, n)
                u = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Bounds check
                u = np.clip(u, lower, upper)
                
                # Evaluation
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= fit[i]:
                    pop[i] = u
                    fit[i] = f_u
                    if f_u < best_val:
                        best_val = f_u
                        best_pos = u.copy()
                        
        return best_pos, best_val
# EVOLVE-BLOCK-END
