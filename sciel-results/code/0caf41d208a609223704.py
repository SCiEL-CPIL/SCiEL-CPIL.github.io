"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Differential Evolution with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        def call_func(x):
            nonlocal best_x, best_f, calls_made
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Multi-Start Differential Evolution
        # DE parameters
        pop_size = 20
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # We need to leave enough budget for NM and ensure we don't exceed max_calls
        # NM in 5D takes roughly 50-100 calls to converge well.
        # Let's allocate 80% of budget to DE, 20% to NM.
        de_budget = int(max_calls * 0.8)
        
        # Initial population
        pop = np.random.uniform(-5, 5, (pop_size, self.dim))
        
        # Evaluate initial population
        for i in range(pop_size):
            if calls_made >= max_calls:
                break
            fx = call_func(pop[i])
        
        # DE Loop
        # We will run DE for a fixed number of generations or until budget/time is up
        # Since DE is population-based, each generation costs pop_size calls.
        # Generations = de_budget // pop_size
        max_generations = max(1, de_budget // pop_size)
        
        for gen in range(max_generations):
            # Check time and budget
            if time.time() > deadline - 10: # 10s margin for NM
                break
            if calls_made + pop_size > max_calls:
                break
                
            # Sort population by fitness (we track best_f, but need sorted pop for DE)
            # To save calls, we don't re-evaluate, but we need fitness values.
            # We must store fitness for current pop.
            # Note: The above loop didn't store fitness for all, only best. 
            # We need to store fitness for all members to perform DE selection.
            # Let's restructure to store fitness array.
            
            # Actually, the previous loop evaluated initial pop. 
            # We need an array of fitnesses.
            # Let's restart the logic slightly to be cleaner and safer.
            # I will re-implement the DE loop properly inside the call.
            
            pass # Placeholder, real logic below

        # --- Corrected Implementation ---
        # Reset state for clean logic
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def call_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Multi-Start DE
        pop_size = 15
        F = 0.8
        CR = 0.9
        
        # Reserve budget for NM (approx 50-100 calls)
        nm_budget = 80
        de_budget = max(10, max_calls - nm_budget)
        
        # Initialize Population
        pop = np.random.uniform(-5, 5, (pop_size, self.dim))
        fit = np.zeros(pop_size)
        
        for i in range(pop_size):
            if calls_made >= max_calls:
                break
            fit[i] = call_func(pop[i])
        
        # DE Loop
        while calls_made + pop_size <= max_calls:
            if time.time() > deadline - 15: # Leave 15s for NM
                break
                
            # Sort by fitness
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if calls_made >= max_calls:
                    break
                    
                # Mutation
                # Standard DE/rand/1
                # Pick 3 distinct random indices different from i
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate
                ft = call_func(trial)
                
                # Selection
                if ft <= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = ft
            
            pop = new_pop
            fit = new_fit
            
            # Early stop if no improvement in last few generations? 
            # Keep simple for now.

        # 2. Nelder-Mead Polish
        # Use the best point found by DE
        if best_x is not None:
            # NM usually needs 50-100 calls for 5D
            # We have remaining budget: max_calls - calls_made
            # Check time
            if time.time() < deadline - 5:
                try:
                    # scipy minimize
                    # options: maxiter is not directly calls, but we can use maxfev
                    # maxfev = remaining calls
                    remaining = max_calls - calls_made
                    if remaining > 10:
                        res = minimize(
                            call_func, 
                            best_x, 
                            method='Nelder-Mead', 
                            options={'maxiter': 100, 'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                        # call_func updates best_x/best_f automatically
                except Exception:
                    pass

        # Ensure we hit exactly 500 calls if possible to maximize AOCC denominator
        # If we have calls left, do random sampling or small steps around best
        while calls_made < max_calls:
            if time.time() > deadline - 1:
                break
            if best_x is None:
                x = np.random.uniform(-5, 5, self.dim)
            else:
                # Small perturbation
                x = best_x + np.random.normal(0, 0.1, self.dim)
            call_func(x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
