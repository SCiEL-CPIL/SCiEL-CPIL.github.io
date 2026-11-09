"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Initial population sampling
        pop_size = max(10, 4 * n)  # Population size
        init_budget = pop_size
        
        # Initialize population uniformly in the box
        population = rng.uniform(-5, 5, (pop_size, n))
        fvals = np.array([func(x) for x in population])
        
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls_made = init_budget
        
        # Phase 2: Differential Evolution
        # DE/rand/1/bin with adaptive F and CR
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        
        # Track convergence for adaptive parameter adjustment
        no_improve = 0
        max_no_improve = 20
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
            
            # Adaptive parameter adjustment
            if no_improve > 0:
                F = max(0.5, F - 0.01)
                CR = min(0.9, CR + 0.02)
            
            # Generate offspring
            offspring = []
            offspring_fvals = []
            
            for i in range(pop_size):
                if time.time() > deadline or calls_made >= total_budget:
                    break
                
                # Select three distinct random individuals
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i:
                        break
                
                # Create mutant vector
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5, 5)
                
                # Binary crossover
                j_rand = rng.integers(0, n)
                trial = population[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluate trial
                trial_f = func(trial)
                calls_made += 1
                
                # Selection: keep better of trial and current
                if trial_f <= fvals[i]:
                    population[i] = trial
                    fvals[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Check for stagnation
            if no_improve >= max_no_improve:
                # Perturb the population to escape local optima
                for i in range(pop_size):
                    population[i] = best_x + 0.5 * rng.standard_normal(n)
                    population[i] = np.clip(population[i], -5, 5)
                    fvals[i] = func(population[i])
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = population[i].copy()
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                F = 0.8
                CR = 0.5
            
            # Check time
            if time.time() > deadline:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
