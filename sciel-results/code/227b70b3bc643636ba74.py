"""Optimizer for BBOB-style f22 using Differential Evolution with Local Polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with Nelder-Mead polishing for 5D multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # DE Parameters
        pop_size = 20
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        f_pop = np.zeros(pop_size)
        
        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Evaluate initial population
        for i in range(pop_size):
            if calls_made >= budget:
                break
            f_pop[i] = eval_point(pop[i])
            
        if best_x is None:
            return np.zeros(dim), np.inf

        # Adaptive F schedule
        f_start = 0.9
        cr = 0.5
        
        # DE Loop
        # Reserve some budget for local search (e.g., 50 calls)
        # If budget is small, skip local search?
        # Let's reserve 50 calls.
        reserve = 50
        while calls_made < budget - reserve:
            # Check if we can afford a full generation
            if calls_made + pop_size > budget - reserve:
                break
            
            # Generate offspring
            new_pop = np.zeros_like(pop)
            new_f = np.zeros(pop_size)
            
            # Decay F
            f_curr = f_start * np.exp(-calls_made / 200.0)
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                # Use rng.integers and ensure distinctness
                indices = rng.integers(0, pop_size, size=3)
                # Ensure distinct and not i
                while len(set(indices)) < 3 or i in indices:
                    indices = rng.integers(0, pop_size, size=3)
                a, b, c = indices
                
                # Mutation: DE/rand/1
                v = pop[a] + f_curr * (pop[b] - pop[c])
                
                # Crossover: Binomial
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                mask = rng.random(dim) < cr
                mask[j_rand] = True
                trial[mask] = v[mask]
                
                # Boundary
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                new_f[i] = eval_point(trial)
                new_pop[i] = trial
                
                # Greedy Selection (1:1)
                if new_f[i] <= f_pop[i]:
                    f_pop[i] = new_f[i]
                    pop[i] = trial
            
        # Local Polishing
        if calls_made < budget:
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'maxfun': budget - calls_made, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass # Keep previous best

        return best_x, best_f
# EVOLVE-BLOCK-END
