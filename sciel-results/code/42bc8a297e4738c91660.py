"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Population size
        N = 15
        # Number of generations
        n_gen = budget // N
        if n_gen < 1:
            n_gen = 1
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls_made = N
        remaining = budget - calls_made
        
        # Adaptive parameters
        F_start = 0.8
        F_end = 0.3
        CR = 0.5
        
        for g in range(n_gen):
            if remaining <= 0:
                break
            
            # Adapt F: start high for exploration, decrease for exploitation
            progress = g / max(n_gen - 1, 1)
            F = F_start + (F_end - F_start) * progress
            
            # Shuffle population for diversity
            perm = rng.permutation(N)
            
            # Evaluate up to 'remaining' individuals this generation
            n_eval_this_gen = min(N, remaining)
            
            for i in range(n_eval_this_gen):
                if remaining <= 0:
                    break
                
                # DE/rand/1/bin mutation
                # Pick three distinct random indices different from i
                other_idx = [j for j in perm if j != i]
                a, b, c = other_idx[:3]
                
                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Binary crossover
                trial = np.empty(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                remaining -= 1
                calls_made += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    
                    # Update best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
