"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Parameters for DE
        pop_size = 20  # Small population for low dim
        F = 0.8        # Mutation factor
        CR = 0.9       # Crossover rate
        
        # Initialize population uniformly in [-4.5, 4.5]
        pop = rng.uniform(-4.5, 4.5, size=(pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        call_count = pop_size
        
        # Track global best
        global_best_idx = np.argmin(fitness)
        global_best_x = pop[global_best_idx].copy()
        global_best_f = fitness[global_best_idx]
        
        # Main loop
        while call_count < budget:
            # Check if we have enough budget for a full generation
            if budget - call_count < pop_size:
                # Do a partial generation or single step
                # For simplicity, if remaining budget is small, just do random perturbations around best
                remaining = budget - call_count
                for _ in range(remaining):
                    # Perturb best
                    perturbation = rng.normal(0, 0.5, dim)
                    x_new = np.clip(global_best_x + perturbation, -5, 5)
                    f_new = func(x_new)
                    call_count += 1
                    if f_new < global_best_f:
                        global_best_f = f_new
                        global_best_x = x_new.copy()
                break

            # Standard DE step
            for i in range(pop_size):
                if call_count >= budget:
                    break
                
                # Select distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluation
                f_trial = func(trial)
                call_count += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = trial.copy()
                        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
