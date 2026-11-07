"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class BudgetExhausted(Exception):
    pass

class Optimizer:
    """DE with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        
        calls_made = 0
        x_best = None
        f_best = np.inf

        def count_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                raise BudgetExhausted()
            if time.time() > deadline:
                raise BudgetExhausted()
            calls_made += 1
            return func(x)

        # Phase 1: Differential Evolution
        N = 20  # Population size
        if self.budget < N:
            # Fallback to simple random search if budget is tiny
            for _ in range(self.budget):
                x = rng.uniform(-5, 5, self.dim)
                fx = count_func(x)
                if fx < f_best:
                    x_best = x
                    f_best = fx
            return x_best, f_best

        # Initialize population
        pop = rng.uniform(-5, 5, (N, self.dim))
        f_pop = np.array([count_func(x) for x in pop])
        
        best_idx = np.argmin(f_pop)
        x_best = pop[best_idx].copy()
        f_best = f_pop[best_idx]

        F = 0.8
        CR = 0.5

        # DE Loop
        while calls_made + N <= self.budget and time.time() < deadline:
            for i in range(N):
                if time.time() > deadline:
                    break
                # Select 3 distinct indices different from i
                candidates = [j for j in range(N) if j != i]
                j, k, l = rng.choice(candidates, 3, replace=False)
                
                v = pop[j] + F * (pop[k] - pop[l])
                
                u = pop[i].copy()
                mask = rng.random(self.dim) < CR
                if not mask.any():
                    mask[rng.integers(self.dim)] = True
                u[mask] = v[mask]
                u = np.clip(u, -5, 5)
                
                try:
                    f_u = count_func(u)
                except BudgetExhausted:
                    return x_best, f_best
                
                if f_u < f_pop[i]:
                    pop[i] = u
                    f_pop[i] = f_u
                    if f_u < f_best:
                        x_best = u.copy()
                        f_best = f_u

        # Phase 2: Nelder-Mead Polish
        if x_best is not None and calls_made < self.budget and time.time() < deadline:
            def wrapped_func(x):
                return count_func(x)
            
            try:
                res = minimize(wrapped_func, x_best, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < f_best:
                    x_best = res.x
                    f_best = res.fun
            except BudgetExhausted:
                pass
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
