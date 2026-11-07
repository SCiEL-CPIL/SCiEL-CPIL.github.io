"""Optimized program for BBOB f21 (Gallagher 101 peaks) with Differential Evolution + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    """
    Portfolio search using Differential Evolution for global exploration
    and L-BFGS-B for local exploitation.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        all_points = []
        all_fits = []
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            x = np.asarray(x, dtype=np.float64)
            # Clip to domain to ensure valid input, though func clips too
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            all_points.append(x.copy())
            all_fits.append(val)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Strategy:
        # 1. Initial random sampling to identify promising regions.
        # 2. Use a simplified Differential Evolution (DE) to explore the global landscape.
        #    DE is robust and requires fewer hyperparameters than CMA-ES for small dims.
        # 3. Periodically trigger local refinement (L-BFGS-B) on the best points found so far.
        #    L-BFGS-B is very efficient for smooth functions and can quickly converge to local minima.
        #    Since f21 is multimodal, we need multiple local refinements from different basins.
        
        # Phase 1: Initial Exploration with Random Sampling + Simple DE
        # We will run a DE-like loop. 
        # Population size N = 10 * dim = 50? That's a lot of calls per generation.
        # Budget is 500. 
        # Let's use a smaller population, e.g., N=20. 
        # Generations: ~25.
        # In each generation, we evaluate N new points (mutations).
        # Additionally, we keep a "hall of fame" of best points.
        
        # However, pure DE might be slow to converge in high dimensions if the step size isn't adapted well.
        # Let's combine:
        # - Start with 20 random points.
        # - Then, for each step, pick the best point, and generate mutations using DE rules (x_best + F*(x_rand1 - x_rand2)).
        # - If a mutation is better, it replaces the worst point in the current "active" set or just gets added to the global best list.
        # - Every K steps, take the top M points from the global best list and run L-BFGS-B on them.
        
        # Let's implement a hybrid:
        # 1. Sample 30 random points. (30 calls)
        # 2. Sort them.
        # 3. Main loop:
        #    - Pick the current global best as the base.
        #    - Generate 10 mutants using DE strategy: base + F * (rand1 - rand2)
        #      where rand1, rand2 are from the current population of best 20 points.
        #    - Evaluate mutants.
        #    - Update global best.
        #    - If stagnation (no improvement in last 20 calls), trigger local refinement on top 3 points.
        
        # Let's be more structured.
        
        # Initial Points
        n_init = 30
        X_init = rng.uniform(-4.5, 4.5, (n_init, dim))
        for x in X_init:
            if n_calls >= budget: break
            wrapped_func(x)
            
        if not all_points:
            return best_x, best_f

        # Sort initial points
        idx = np.argsort(all_fits)
        # Keep a "population" of the best 20 points for DE mutation
        pop_size = 20
        if len(idx) >= pop_size:
            pop_indices = idx[:pop_size]
            population = [all_points[i].copy() for i in pop_indices]
            pop_fits = [all_fits[i] for i in pop_indices]
        else:
            population = [all_points[i].copy() for i in idx]
            pop_fits = [all_fits[i] for i in idx]

        # DE Parameters
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate (though we might just use full vector mutation for simplicity in 5D)
        
        stagnation_counter = 0
        last_best_f = best_f
        
        # Local Refinement Function
        def local_refine(x0, max_func_calls=20):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget - 5:
                return
            
            # L-BFGS-B requires bounds
            bounds = [(-5.0, 5.0)] * dim
            
            # We need a wrapper that counts calls and stops if budget is exceeded
            local_calls = 0
            def local_func(x):
                nonlocal local_calls
                if local_calls >= max_func_calls or n_calls >= budget:
                    return 1e18
                local_calls += 1
                return wrapped_func(x)
            
            try:
                res = minimize(local_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 10, 'ftol': 1e-6, 'gtol': 1e-4})
                # Note: scipy minimize might make more calls than expected if it doesn't stop cleanly
                # But we capped it in local_func.
            except Exception:
                pass
            
            # Update global best if local found something better (wrapped_func already did)
            # But we need to update the population if local found a new best
            if best_f < last_best_f:
                stagnation_counter = 0
                last_best_f = best_f
                # Add the new best to the front of the population
                # Replace the worst in population
                if population:
                    worst_idx = np.argmax(pop_fits)
                    population[worst_idx] = best_x.copy()
                    pop_fits[worst_idx] = best_f
                    # Re-sort population? 
                    # For simplicity, just keep it. Or re-sort.
                    sort_idx = np.argsort(pop_fits)
                    population = [population[i] for i in sort_idx]
                    pop_fits = [pop_fits[i] for i in sort_idx]

        # Main Loop
        # We will iterate until budget is exhausted or time is up.
        # Each iteration generates a few mutants.
        # Let's generate 5 mutants per iteration.
        mutants_per_iter = 5
        
        while n_calls < budget and time.time() < deadline - 10:
            
            # Check for stagnation
            if best_f < last_best_f:
                stagnation_counter = 0
                last_best_f = best_f
            else:
                stagnation_counter += 1
            
            # If stagnation is high, trigger local refinement on top 2 points
            if stagnation_counter > 30:
                # Refine the current best
                if population:
                    local_refine(population[0], max_func_calls=30)
                    # Refine the second best
                    if len(population) > 1:
                        local_refine(population[1], max_func_calls=30)
                
                # Reset stagnation counter after refinement
                stagnation_counter = 0
                last_best_f = best_f
                
                # If budget is still low, continue
                if n_calls >= budget:
                    break
            
            # Generate Mutants
            for _ in range(mutants_per_iter):
                if n_calls >= budget:
                    break
                
                # Pick base (best point)
                base = population[0]
                
                # Pick two random distinct points from population
                if len(population) < 3:
                    # If population is too small, just use random jumps
                    mutant = base + rng.normal(0, 0.5, dim)
                else:
                    # Pick two different indices
                    i1, i2 = rng.choice(len(population), 2, replace=False)
                    rand1 = population[i1]
                    rand2 = population[i2]
                    
                    # DE Mutation
                    mutant = base + F * (rand1 - rand2)
                    
                    # Optional: Add some random noise to help escape local minima
                    mutant += rng.normal(0, 0.1, dim)
                
                # Clip to bounds
                mutant = np.clip(mutant, -5.0, 5.0)
                
                # Evaluate
                f_mutant = wrapped_func(mutant)
                
                # Update population if mutant is better than the worst
                if population and f_mutant < pop_fits[-1]:
                    # Replace worst
                    population[-1] = mutant.copy()
                    pop_fits[-1] = f_mutant
                    # Re-sort
                    sort_idx = np.argsort(pop_fits)
                    population = [population[i] for i in sort_idx]
                    pop_fits = [pop_fits[i] for i in sort_idx]
            
            # Periodically trigger local refinement even without high stagnation?
            # Maybe every 50 calls, refine the top 1 point.
            if n_calls % 50 == 0 and population:
                local_refine(population[0], max_func_calls=20)

        return best_x, best_f
# EVOLVE-BLOCK-END
