"""Optimizer using Differential Evolution (DE) with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # DE Parameters
        NP = 10  # Population size
        CR = 0.5  # Crossover rate
        F = 0.8   # Mutation factor
        
        # Reserve budget for local search
        # 10% for local search, 90% for DE
        local_budget = max(10, budget // 10)
        de_budget = budget - local_budget
        
        # Initialize population
        # Use a fixed seed for reproducibility within the run? 
        # The evaluator might set a global seed. 
        # Using np.random directly relies on global state.
        # Let's use a local RNG for consistency.
        rng = np.random.default_rng(42)
        
        pop = rng.uniform(-5, 5, (NP, n))
        
        # Evaluate initial population
        f_pop = np.array([func(x) for x in pop])
        calls = NP
        
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        # DE Loop
        max_gens = (de_budget - calls) // NP
        
        for g in range(max_gens):
            if calls + NP > de_budget:
                break
                
            for i in range(NP):
                # Select 3 distinct indices for mutation
                # Ensure they are distinct from each other
                # And preferably distinct from i to avoid self-cancellation
                while True:
                    idxs = rng.choice(NP, 3, replace=False)
                    if i not in idxs:
                        break
                
                r1, r2, r3 = idxs
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binomial Crossover
                j_rand = rng.integers(0, n)
                cross_mask = rng.random(n) < CR
                cross_mask[j_rand] = True
                
                trial_x = np.where(cross_mask, v, pop[i])
                trial_x = np.clip(trial_x, -5, 5)
                
                f_trial = func(trial_x)
                calls += 1
                
                # Selection
                if f_trial <= f_pop[i]:
                    pop[i] = trial_x
                    f_pop[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial_x.copy()

        # Local Refinement: Simple Hill Climbing
        if local_budget > 0 and calls < budget:
            x = best_x.copy()
            f_x = best_f
            step_size = 1.0
            
            for _ in range(local_budget):
                if calls >= budget:
                    break
                
                # Random direction
                dir_vec = rng.standard_normal(n)
                norm = np.linalg.norm(dir_vec)
                if norm > 0:
                    dir_vec /= norm
                else:
                    continue
                    
                x_new = x + step_size * dir_vec
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls += 1
                
                if f_new < f_x:
                    x = x_new
                    f_x = f_new
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                else:
                    step_size *= 0.7
                    if step_size < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
