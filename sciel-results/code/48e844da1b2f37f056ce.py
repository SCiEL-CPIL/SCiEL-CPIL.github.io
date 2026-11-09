import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        NP = 10  # Population size
        F = 0.6  # Mutation factor
        CR = 0.8  # Crossover rate
        
        if budget < NP:
            points = rng.uniform(-5, 5, (budget, n))
            fits = np.array([func(x) for x in points])
            best_idx = np.argmin(fits)
            return points[best_idx], fits[best_idx]

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, n))
        fits = np.array([func(x) for x in pop])
        evals_used = NP
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        # Precompute indices for faster deletion
        all_indices = np.arange(NP)
        
        # Main Loop
        while evals_used < budget:
            if evals_used + NP > budget:
                break
            
            for i in range(NP):
                # Get 3 distinct random indices excluding i
                indices_no_i = np.delete(all_indices, i)
                r1, r2, r3 = rng.choice(indices_no_i, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(n)
                mask = rng.random(n) < CR
                mask[j_rand] = True
                trial = np.where(mask, v, pop[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                evals_used += 1
                
                # Selection
                if f_trial <= fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
