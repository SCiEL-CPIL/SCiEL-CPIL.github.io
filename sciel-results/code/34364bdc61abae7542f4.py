import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Dense grid + random initialization
        # Katsuura has regular structure; sample densely
        initial_points = []
        
        # Coarse grid in [-5, 5]^dim with spacing ~1.0
        grid_vals = np.linspace(-5, 5, 6)  # 6 points per dim = 6^5 = 7776, too many
        # Use a sparser structured set + random
        for _ in range(20):
            initial_points.append(rng.uniform(-5, 5, dim))
        
        # Focused region where minimum lies: [-4, 4]^5
        for _ in range(30):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Center
        initial_points.append(np.zeros(dim))
        
        # Evaluate initial points
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 15.0:
                break
            x0 = clip(x0)
            evaluate(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: Nelder-Mead refinement from best point
        if calls_made < budget and remaining_time() > 15.0:
            local_budget = min(budget - calls_made, 150)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 10.0
                return evaluate(x)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': local_budget, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass

        # Phase 3: Differential Evolution (DE/rand/1) with population of 40
        remaining_calls = budget - calls_made
        if remaining_calls > 50 and remaining_time() > 20.0:
            pop_size = 40
            # Initialize population: mix of random, near-best, and structured
            population = []
            fitnesses = []
            
            # Seed with best found
            population.append(best_x.copy())
            fitnesses.append(best_f)
            
            # Add diverse points
            for i in range(pop_size - 1):
                if i < 8:
                    # Near best
                    x = clip(best_x + 0.3 * rng.standard_normal(dim))
                elif i < 20:
                    # Mid-range
                    x = clip(rng.uniform(-4, 4, dim))
                else:
                    # Full range
                    x = clip(rng.uniform(-5, 5, dim))
                population.append(x)
                fitnesses.append(evaluate(x))
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]
            
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0].copy()
            
            # DE/rand/1 loop with adaptive parameters
            F = 0.7  # scaling factor
            CR = 0.8  # crossover rate
            max_generations = max(1, (budget - calls_made - pop_size) // pop_size)
            
            for gen in range(max_generations):
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                
                # Update best from population
                if fitnesses[0] < best_f:
                    best_f = fitnesses[0]
                    best_x = population[0].copy()
                
                new_pop = []
                new_fit = []
                
                for i in range(pop_size):
                    if calls_made >= budget or remaining_time() < 3.0:
                        new_pop.append(population[i])
                        new_fit.append(fitnesses[i])
                        continue
                    
                    # Select three distinct random indices different from i
                    r1, r2, r3 = i
                    while r1 == i:
                        r1 = rng.integers(0, pop_size)
                    while r2 == i or r2 == r1:
                        r2 = rng.integers(0, pop_size)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = rng.integers(0, pop_size)
                    
                    # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                    v = population[r1] + F * (population[r2] - population[r3])
                    v = clip(v)
                    
                    # Crossover: binary crossover with CR
                    trial = np.empty(dim)
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]
                    
                    trial = clip(trial)
                    f_trial = evaluate(trial)
                    
                    # Selection
                    if f_trial <= fitnesses[i]:
                        new_pop.append(trial)
                        new_fit.append(f_trial)
                    else:
                        new_pop.append(population[i])
                        new_fit.append(fitnesses[i])
                
                # Sort new population
                order = np.argsort(new_fit)
                population = [new_pop[i] for i in order]
                fitnesses = [new_fit[i] for i in order]
                
                # Inject best into population periodically to prevent loss
                if gen % 5 == 0 and fitnesses[0] > best_f:
                    population[0] = best_x.copy()
                    fitnesses[0] = best_f
                    order = np.argsort(fitnesses)
                    population = [population[i] for i in order]
                    fitnesses = [fitnesses[i] for i in order]
                
                # Adapt F: reduce if stagnating, increase if exploring
                spread = fitnesses[-1] - fitnesses[0]
                if gen > 0 and spread < 1e-6:
                    F = max(0.3, F * 0.95)
                    CR = min(0.95, CR * 1.05)
                elif gen > 0 and spread > 10.0:
                    F = min(0.9, F * 1.05)
                    CR = max(0.7, CR * 0.95)
        
        # Phase 4: Final high-precision polish on best
        if calls_made < budget and remaining_time() > 5.0:
            local_budget = min(budget - calls_made, 100)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 10.0
                return evaluate(x)
            
            try:
                # First pass: Nelder-Mead
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': min(local_budget, 60), 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass
            
            # Second pass: tighter Nelder-Mead if time/calls remain
            if calls_made < budget and remaining_time() > 3.0:
                remaining_local = min(budget - calls_made, 40)
                if remaining_local > 5:
                    local_calls2 = [0]
                    
                    def local_func2(x):
                        local_calls2[0] += 1
                        if local_calls2[0] > remaining_local:
                            return best_f + 10.0
                        return evaluate(x)
                    
                    try:
                        res2 = minimize(local_func2, best_x, method='Nelder-Mead',
                                     options={'maxiter': remaining_local, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res2.fun < best_f:
                            best_f = float(res2.fun)
                            best_x = clip(res2.x)
                    except Exception:
                        pass

        return best_x, best_f
