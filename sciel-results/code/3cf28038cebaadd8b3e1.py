"""Optimized program for BBOB f21 (Gallagher 101 peaks) with DE-then-L-BFGS-B hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer:
    1. Phase 1 (50% budget): Differential Evolution (DE) for global exploration.
       DE is robust to multi-modality and does not require gradient information.
    2. Phase 2 (50% budget): L-BFGS-B local refinement from the best DE candidate.
       L-BFGS-B is a quasi-Newton method that converges quickly to local minima
       when initialized in the correct basin.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit. Leave 5s for cleanup/return.
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Fixed seed for reproducibility in the evaluator context if needed, 
        # but random is generally fine. Using a fixed seed helps debugging.
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            x = np.asarray(x, dtype=np.float64)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Differential Evolution (DE) ---
        # DE parameters
        # Population size: typically 10-15 * dim
        pop_size = 15 * dim 
        # Ensure we don't exceed budget in phase 1
        # Each generation uses pop_size calls.
        # We want to use ~50% of budget for DE.
        # max_generations = (budget * 0.5) // pop_size
        # But we also need to handle the initial population evaluation.
        
        # Let's reserve budget carefully.
        # Phase 1 budget: 50% of total
        phase1_budget = int(budget * 0.5)
        # If pop_size is too large for phase1_budget, reduce it
        if phase1_budget < pop_size:
             pop_size = max(10, phase1_budget // 2)
        
        # Initialize population
        population = rng.uniform(-5, 5, size=(pop_size, dim))
        fitness = np.array([wrapped_func(ind) for ind in population])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_f_de = fitness[best_idx]
        best_x_de = population[best_idx].copy()
        
        # DE Parameters
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Number of generations we can afford
        # Each generation: pop_size new evaluations
        # We have already used pop_size evaluations for init.
        remaining_phase1_calls = phase1_budget - pop_size
        
        n_generations = remaining_phase1_calls // pop_size
        
        for gen in range(n_generations):
            if time.time() > deadline - 10:
                break
            
            # Update global best
            if best_f_de < best_f:
                best_f = best_f_de
                best_x = best_x_de.copy()

            # Generate offspring
            for i in range(pop_size):
                # Check time and budget
                if time.time() > deadline - 10 or n_calls >= phase1_budget:
                    break
                
                # Select three distinct random indices different from i
                a, b, c = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
                
                # Mutation: DE/rand/1
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f_de:
                        best_f_de = f_trial
                        best_x_de = trial.copy()

        # --- Phase 2: L-BFGS-B Local Refinement ---
        # Use the best point found by DE as the starting point for L-BFGS-B.
        # L-BFGS-B is efficient for smooth functions.
        
        # We might want to try a few local refinements from top DE candidates
        # if budget allows, but let's stick to the single best one for simplicity 
        # and to maximize the budget for the refinement itself if it's slow.
        
        # Actually, to be safe against getting stuck in a local minimum that isn't the global one,
        # we can try refining the top 3 candidates from DE if we have time.
        
        # Sort DE results
        sorted_indices = np.argsort(fitness)
        top_k = min(3, pop_size)
        
        # Reserve time for L-BFGS-B
        # L-BFGS-B can take many iterations. 
        # We have budget - n_calls remaining.
        
        for k in range(top_k):
            if n_calls >= budget:
                break
            if time.time() > deadline - 5:
                break
                
            start_x = population[sorted_indices[k]].copy()
            
            # Define objective for scipy
            def objective(x):
                return wrapped_func(x)
            
            # L-BFGS-B
            try:
                res = minimize(
                    objective, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

        # Final check: ensure we haven't exceeded budget
        # The wrapped_func already guards against exceeding budget.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
