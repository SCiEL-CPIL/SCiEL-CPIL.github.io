import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, n))
        fit = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        evals_used = pop_size
        
        # Pre-generate indices for speed? No, random is fine.
        while evals_used + pop_size <= self.budget:
            new_pop = []
            new_fit = []
            
            for i in range(pop_size):
                # Efficient selection of 3 distinct indices different from i
                # Using rng.choice is slow. Use manual selection.
                indices = [i]
                while len(indices) < 4:
                    r = rng.integers(pop_size)
                    if r not in indices:
                        indices.append(r)
                r1, r2, r3 = indices[1], indices[2], indices[3]
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                evals_used += 1
                
                # Selection
                if f_trial < fit[i]:
                    new_pop.append(trial)
                    new_fit.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop.append(pop[i])
                    new_fit.append(fit[i])
            
            pop = np.array(new_pop)
            fit = np.array(new_fit)
            
        # Use remaining budget if any
        while evals_used < self.budget:
            # Simple local search on best_x? Or just perturb?
            # Perturb best_x with small noise
            noise = rng.standard_normal(n) * 0.1
            trial = np.clip(best_x + noise, -5, 5)
            f_trial = func(trial)
            evals_used += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = trial.copy()
                
        return best_x, best_f
