"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Differential Evolution with Nelder-Mead polish for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
            
        def de_search(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run Differential Evolution from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            # Population size: standard is 5-10 * dim. For dim=5, let's use 40 for diversity.
            pop_size = 40
            # Differential weight
            F = 0.8
            # Crossover rate
            CR = 0.9
            
            # Initialize population
            # x0 is one of the members (the best one found so far)
            population = np.zeros((pop_size, dim))
            fitness = np.zeros(pop_size)
            
            # Seed with x0 and random points
            population[0] = x0
            fitness[0] = f0
            
            for i in range(1, pop_size):
                population[i] = rng.uniform(-5, 5, dim)
                fitness[i] = func(population[i])
                calls_used += 1
                
            # Track global best
            idx_best = np.argmin(fitness)
            best_x_local = population[idx_best].copy()
            best_f_local = fitness[idx_best]
            
            # Main loop
            # Each generation costs pop_size evaluations (approx, actually we evaluate trial vectors)
            # We need to be careful with the budget.
            # In DE, we generate pop_size trial vectors and evaluate them.
            # So each generation consumes ~pop_size calls.
            
            while calls_used + pop_size < max_calls and time_left() > 1.0:
                new_population = population.copy()
                new_fitness = fitness.copy()
                
                for i in range(pop_size):
                    # Select two distinct random indices different from i
                    r1, r2 = rng.choice(pop_size - 1, size=2, replace=False)
                    # Ensure r1, r2 are not i
                    # A simpler way: sample from all, retry if same
                    while r1 == i:
                        r1 = rng.integers(pop_size)
                    while r2 == i or r2 == r1:
                        r2 = rng.integers(pop_size)
                        
                    # Mutation: v = x_r1 + F * (x_r2 - x_i)
                    v = population[r1] + F * (population[r2] - population[i])
                    # Clip to bounds
                    v = np.clip(v, -5, 5)
                    
                    # Crossover
                    j_rand = rng.integers(dim)
                    trial = population[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    # Ensure trial is within bounds (though v was clipped, crossover might not mix well if bounds are tight, 
                    # but since we clipped v and population is in bounds, trial is in bounds)
                    
                    # Evaluation
                    f_trial = func(trial)
                    calls_used += 1
                    
                    # Selection
                    if f_trial <= fitness[i]:
                        new_population[i] = trial
                        new_fitness[i] = f_trial
                        if f_trial < best_f_local:
                            best_f_local = f_trial
                            best_x_local = trial.copy()
                            
                population = new_population
                fitness = new_fitness
                
                # Convergence check: if best hasn't improved in many generations, stop?
                # Or if sigma is too small? DE doesn't have sigma.
                # Let's just run until budget or time.
                
            return best_x_local, best_f_local, calls_used

        def nelder_mead_polish(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run Nelder-Mead from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            # Scipy's minimize uses a specific interface. 
            # We need to wrap func to count calls and respect deadline.
            calls_used = 1  # f0 is already known
            x_current = x0.copy()
            f_current = f0
            
            # Scipy Nelder-Mead implementation
            # We pass a function that increments a counter
            counter = [0]
            def wrapped_func(x):
                nonlocal calls_used
                if time_left() < 0.1:
                    raise Exception("Time's up")
                fx = func(x)
                calls_used += 1
                counter[0] += 1
                return fx
            
            # Limit iterations to avoid hanging
            # maxiter in scipy corresponds to number of function calls roughly? No, it's iterations.
            # For dim=5, NM is fast. 
            # We'll use a safe maxiter.
            try:
                result = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': max(10, max_calls - calls_used), 'xatol': 1e-6, 'fatol': 1e-6})
                # result.fun is the best found
                if result.fun < f_current:
                    f_current = result.fun
                    x_current = result.x
            except Exception:
                pass
            
            return x_current, f_current, calls_used

        # Phase 1: Initial Sampling
        # Use a small number of random points to seed DE or just start DE directly?
        # DE initializes its own population. 
        # Let's just start DE directly with a random seed point.
        
        # However, to ensure we have a good starting point for DE, 
        # let's do 10 random samples to find a reasonable x0.
        n_init = 10
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls_made += 1
        
        for i in range(n_init - 1):
            if calls_made >= budget or time_left() < 1.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < f0:
                f0 = fx
                x0 = x.copy()
                
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
            
        # Phase 2: Differential Evolution
        # Allocate 80% of remaining budget to DE
        remaining = budget - calls_made
        de_budget = int(remaining * 0.8)
        
        x_de, f_de, used_de = de_search(func, x0, f0, de_budget, deadline, dim, rng)
        calls_made += used_de
        
        if f_de < best_f:
            best_f = f_de
            best_x = x_de.copy()
            
        # Phase 3: Nelder-Mead Polish
        # Use the remaining 20% for local search
        remaining = budget - calls_made
        nm_budget = remaining
        
        if nm_budget > 10 and time_left() > 5.0:
            x_nm, f_nm, used_nm = nelder_mead_polish(func, best_x, best_f, nm_budget, deadline, dim, rng)
            calls_made += used_nm
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
                
        return best_x, best_f

# EVOLVE-BLOCK-END
