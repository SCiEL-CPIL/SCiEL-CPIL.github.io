import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using Differential Evolution for global search 
    followed by (1+1)-ES for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        calls = 0
        max_calls = self.budget
        
        # Helper to evaluate and track calls
        def eval_func(x):
            nonlocal calls
            if calls >= max_calls:
                return float('inf') 
            calls += 1
            return func(x)

        # Phase 1: Differential Evolution
        pop_size = 15
        # Allocate 80% of budget to DE
        de_budget = int(max_calls * 0.8)
        
        # Initialize Population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-self.bound, self.bound, (pop_size, self.dim))
        fit = np.array([eval_func(x) for x in pop])
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        F = 0.8 # Mutation factor
        CR = 0.9 # Crossover rate
        
        # DE Loop
        # We run until we have used de_budget calls
        while calls < de_budget:
            # Time check
            if time.time() - start_time > 280:
                break
            
            for i in range(pop_size):
                if calls >= de_budget:
                    break
                if time.time() - start_time > 280:
                    break
                
                # Select 3 distinct indices
                # Using choice with replace=False is efficient
                a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]
                
                # Mutation
                v = a + F * (b - c)
                v = np.clip(v, -self.bound, self.bound)
                
                # Crossover
                r = rng.random(self.dim)
                j_rand = rng.integers(0, self.dim)
                u = np.where(r < CR, v, pop[i])
                u[j_rand] = v[j_rand] # Ensure at least one gene from mutant
                
                # Evaluation
                f_u = eval_func(u)
                
                # Selection
                if f_u < fit[i]:
                    pop[i] = u
                    fit[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
        # Phase 2: Local Refinement (Nelder-Mead)
        remaining = max_calls - calls
        if remaining > 0:
            # Define a wrapper for minimize that respects the budget
            def minimize_func(x):
                return eval_func(x)
            
            try:
                result = minimize(minimize_func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass # If minimize fails, keep best from DE

        return best_x, best_f
