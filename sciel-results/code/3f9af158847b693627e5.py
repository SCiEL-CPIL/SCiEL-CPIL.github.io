"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Improved Differential Evolution with adaptive parameters and better initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters - larger population for better exploration
        pop_size = 20
        rng = np.random.default_rng(42)
        
        # Initialize population with stratified sampling for better coverage
        # Use Latin hypercube-like initialization
        population = np.zeros((pop_size, n))
        for i in range(pop_size):
            for j in range(n):
                # Stratify each dimension
                lower = -5 + (10 * i) / pop_size
                upper = -5 + (10 * (i + 1)) / pop_size
                population[i, j] = rng.uniform(lower, upper)
        
        f_values = np.array([func(x) for x in population])
        calls_used = pop_size
        
        global_best_idx = np.argmin(f_values)
        global_best_x = population[global_best_idx].copy()
        global_best_f = f_values[global_best_idx]
        
        # Adaptive DE loop
        # Start with larger mutation, decrease over time
        F_start = 0.9
        F_end = 0.4
        CR_start = 0.2
        CR_end = 0.7
        
        no_improve_count = 0
        best_f_history = global_best_f
        
        while calls_used < budget:
            # Adaptive parameters based on progress
            progress = calls_used / budget
            F = F_start + (F_end - F_start) * progress
            CR = CR_start + (CR_end - CR_start) * progress
            
            # Check if we have enough budget for a full generation
            if budget - calls_used < pop_size:
                # Use remaining budget for single-point mutations around best
                while calls_used < budget:
                    # Local refinement around global best
                    v = global_best_x + F * rng.normal(0, 0.5, size=n)
                    v = np.clip(v, -5, 5)
                    f_trial = func(v)
                    calls_used += 1
                    
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = v.copy()
                break
            
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binomial
                j_rand = rng.integers(0, n)
                trial = np.copy(population[i])
                
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1
                
                # Update global best
                if f_trial < global_best_f:
                    global_best_f = f_trial
                    global_best_x = trial.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Selection
                if f_trial <= f_values[i]:
                    population[i] = trial
                    f_values[i] = f_trial
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
