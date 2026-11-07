import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        best_x_global = None
        best_f_global = np.inf
        calls_used = 0
        
        rng = np.random.default_rng(123)
        
        # Multi-start strategy
        # 3 restarts
        # Initial points: 0, random, random
        initial_points = [
            np.zeros(n),
            rng.uniform(-5, 5, n),
            rng.uniform(-5, 5, n)
        ]
        
        # Allocate budget
        # Give slightly more to the first one? Or equal?
        # Equal is safer.
        num_restarts = len(initial_points)
        budget_per_restart = total_budget // num_restarts
        
        for i in range(num_restarts):
            if calls_used >= total_budget:
                break
            if time.time() > deadline:
                break
            
            x0 = initial_points[i]
            current_budget = min(budget_per_restart, total_budget - calls_used)
            
            if current_budget < 20:
                # Not enough for a meaningful DE run
                # Just evaluate x0 and nearby?
                # Or skip
                continue
            
            x_best, f_best, calls_made = self._run_de(func, x0, current_budget, n, rng)
            calls_used += calls_made
            
            if f_best < best_f_global:
                best_f_global = f_best
                best_x_global = x_best
        
        # If we have spare budget, do a local polish?
        # Or just return
        if best_x_global is None:
            # Fallback
            best_x_global = np.zeros(n)
            best_f_global = func(best_x_global)
            calls_used += 1

        return best_x_global, best_f_global

    def _run_de(self, func, x0, budget, n, rng):
        # Differential Evolution
        # Population size
        NP = 10  # Small for 500 budget
        if NP * 2 > budget:
            NP = max(5, budget // 5)
            
        # Parameters
        F = 0.8
        CR = 0.9
        
        # Initialize population
        # One individual is x0, rest are random around x0 or random in box
        pop = np.empty((NP, n))
        pop[0] = x0
        # Spread the rest
        # Use a wide range to explore
        for i in range(1, NP):
            pop[i] = rng.uniform(-5, 5, n)
            
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        calls = NP
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        stagnation_count = 0
        patience = 5 # Generations without improvement
        
        while calls < budget:
            if time.time() > 290 + time.time() - time.time(): # Placeholder, use absolute check outside
                break
                
            # Check time more explicitly
            # We don't have start_time here, but we can pass it or just rely on the outer loop check
            # Better to check inside if we have a deadline
            
            # Generate trial vectors
            improved = False
            
            for i in range(NP):
                if calls >= budget:
                    break
                    
                # Select two distinct random indices
                r1, r2 = rng.choice(NP, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(NP, 2, replace=False)
                    
                # Differential variation
                v = pop[r1] + F * (pop[r2] - pop[r1])
                
                # Clip to box
                v = np.clip(v, -5, 5)
                
                # Evaluate
                f_v = func(v)
                calls += 1
                
                # Selection
                if f_v <= fitness[i]:
                    pop[i] = v
                    fitness[i] = f_v
                    improved = True
                    
                    if f_v < best_f:
                        best_f = f_v
                        best_x = v.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    # If no improvement in this individual, stagnation count might increase?
                    # Standard DE stagnation is usually per generation
                    pass
            
            # Check stagnation per generation
            if not improved:
                stagnation_count += 1
            else:
                # If best improved, reset
                if best_f < np.min(fitness) + 1e-9: # Best is still the min
                    pass 
                # Actually, if best_f didn't change in this generation, stagnation should count
                # Let's refine: track best_f before and after generation
            
            # Simpler stagnation check:
            # If best_f hasn't improved for 'patience' generations, restart or stop?
            # Since we are in a sub-run, we can just stop this sub-run if stuck.
            
            # Let's use a simpler check:
            # If no individual improved in the last generation, stagnation_count += 1
            # Else stagnation_count = 0
            
            if stagnation_count > patience:
                break
        
        return best_x, best_f, calls
