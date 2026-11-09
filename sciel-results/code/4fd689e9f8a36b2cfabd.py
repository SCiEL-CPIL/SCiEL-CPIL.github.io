import numpy as np
import time
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        pop_size = 10
        cr = 0.5
        f = 0.5
        
        # Initial population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(x) for x in pop])
        calls += pop_size
        idx_best = np.argmin(fit)
        best_f = fit[idx_best]
        best_x = pop[idx_best].copy()
        
        stagnant_count = 0
        
        while calls < budget:
            # Check time/budget
            if time.time() > start_time + 280: break
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if calls >= budget: break
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                # DE/1/bin
                v = pop[idx_best] + f * (pop[r1] - pop[r2])
                # Crossover
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < cr or j == j_rand:
                        offspring[i][j] = v[j]
                    else:
                        offspring[i][j] = pop[i][j]
                offspring[i] = np.clip(offspring[i], -5, 5)
            
            # Evaluate offspring
            off_fit = np.array([func(x) for x in offspring])
            calls += len(offspring)
            
            # Selection
            new_pop = []
            new_fit = []
            for i in range(pop_size):
                if off_fit[i] < fit[i]:
                    new_pop.append(offspring[i])
                    new_fit.append(off_fit[i])
                    if off_fit[i] < best_f:
                        best_f = off_fit[i]
                        best_x = offspring[i].copy()
                        idx_best = i
                        stagnant_count = 0
                else:
                    new_pop.append(pop[i])
                    new_fit.append(fit[i])
            
            pop = np.array(new_pop)
            fit = np.array(new_fit)
            
            # Update best index in pop
            idx_best = np.argmin(fit)
            
            # Stagnation check
            # If best_f hasn't changed, increment counter
            # (Simplified: if idx_best changed, it's not stagnant)
            # Actually, track best_f from previous iteration
            
            # Restart logic
            if stagnant_count > 20: # Tune this
                # Restart population
                # Keep best, randomize others
                pop[0] = best_x
                fit[0] = best_f
                for i in range(1, pop_size):
                    if rng.random() < 0.5:
                        pop[i] = rng.uniform(-5, 5, dim)
                    else:
                        pop[i] = best_x + rng.normal(0, 1, dim)
                        pop[i] = np.clip(pop[i], -5, 5)
                    fit[i] = func(pop[i])
                    calls += 1
                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = pop[i].copy()
                idx_best = np.argmin(fit)
                stagnant_count = 0
            else:
                # Check if best improved
                if fit[idx_best] < best_f - 1e-10:
                     stagnant_count = 0
                else:
                     stagnant_count += 1
                     
        return best_x, best_f
