import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # For dim=5, population size of 10-15 is reasonable
        pop_size = 12
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # State
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial population: uniform in [-5, 5]^dim
        # Actually, let's use a slightly smaller range initially to avoid wasting evals on corners
        # But the function clips, so it's safe.
        # Let's use [-4, 4] as the minimizer is known to be in [-4, 4]^5
        # Wait, the problem says "minimizer lies in [-4, 4]^5".
        # So initializing in [-4, 4] is a good idea.
        
        pop = np.random.uniform(-4, 4, size=(pop_size, dim))
        pop_f = np.array([evaluate(pop[i]) for i in range(pop_size)])
        
        # Sort initial population
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        # DE Loop
        generation = 0
        stagnant = 0
        max_stagnant = 20  # Restart or mutate if no improvement for 20 generations
        
        while calls < budget and time_ok():
            # Generate offspring
            new_pop = []
            new_f = []
            
            for i in range(pop_size):
                if not time_ok() or calls >= budget:
                    break
                    
                # Select 3 distinct parents different from i
                p1, p2, p3 = np.random.choice(
                    [j for j in range(pop_size) if j != i], 
                    size=3, 
                    replace=False
                )
                
                # Differential mutation
                v = pop[p1] + F * (pop[p2] - pop[p3])
                
                # Binomial crossover
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial = v[j]
                    else:
                        trial = pop[i][j]
                    # Clip to [-5, 5]
                    trial = np.clip(trial, -5, 5)
                    
                # Evaluate
                f_trial = evaluate(trial)
                new_pop.append(trial)
                new_f.append(f_trial)
                
                # Selection
                if f_trial < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
            
            if len(new_pop) < pop_size:
                break
                
            # Re-sort population
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]
            
            generation += 1
            
            # Stagnation check
            if pop_f[0] >= best_f:  # This is always true since best_f is min(pop_f)
                pass # Logic above updates best_f, so check improvement
            
            # Actually, let's track the best fitness of the population
            current_best = pop_f[0]
            if abs(current_best - best_f) < 1e-12:
                stagnant += 1
            else:
                stagnant = 0
            
            if stagnant > max_stagnant:
                # Random restart: re-seed a few individuals
                # Replace worst individuals with random ones
                n_replace = pop_size // 2
                for i in range(n_replace, pop_size):
                    pop[i] = np.random.uniform(-4, 4, size=dim)
                    pop_f[i] = evaluate(pop[i])
                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]
                stagnant = 0
        
        # Final refinement with a local optimizer?
        # With 500 calls, we might have used ~480 in DE.
        # We have ~20 calls left.
        # Let's do a few steps of Nelder-Mead or just simple hill climbing on the best point.
        # Given the budget constraint, a simple local search is safer.
        
        if best_x is not None and calls < budget and time_ok():
            # Simple local search (Nelder-Mead lite or coordinate descent)
            # Let's use a small Nelder-Mead
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead from scipy if available, it's robust
                # But scipy might be slow to import? It's already imported if we use it.
                # Let's stick to numpy for safety.
                
                # Coordinate descent
                x = best_x.copy()
                f_x = best_f
                step = 0.5
                for _ in range(10):  # Limit steps to save budget
                    if not time_ok() or calls >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        # Try positive step
                        x_pos = x.copy()
                        x_pos[d] += step
                        f_pos = evaluate(x_pos)
                        if f_pos < f_x:
                            x = x_pos
                            f_x = f_pos
                            improved = True
                        else:
                            # Try negative step
                            x_neg = x.copy()
                            x_neg[d] -= step
                            f_neg = evaluate(x_neg)
                            if f_neg < f_x:
                                x = x_neg
                                f_x = f_neg
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break
            except:
                pass

        return best_x, best_f
