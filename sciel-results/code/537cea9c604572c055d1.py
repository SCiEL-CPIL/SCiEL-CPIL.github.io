"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Differential Evolution for global search, 
    Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        # Phase 1: Differential Evolution (DE)
        # Parameters tuned for 5D, budget 500
        # Population size: typically 5-10 * dim. Let's use 10 * dim = 50.
        # Generations: (budget - pop_size) / pop_size approx.
        # But we need to leave budget for local search.
        # Let's use 300 calls for DE, 200 for local.
        
        pop_size = 30
        # Initial population
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fit = np.array([evaluate(x) for x in pop])
        
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        best_idx = np.argmin(fit)
        global_best_x = pop[best_idx].copy()
        global_best_fx = fit[best_idx]
        
        no_improve = 0
        max_no_improve = 10
        
        while calls_made < 300 and time.time() < deadline:
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if calls_made >= 300 or time.time() > deadline:
                    break
                
                # Select three distinct indices different from i
                indices = np.random.choice(pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(pop_size, 3, replace=False)
                
                a, b, c = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                
                # Mutation
                v = a + F * (b - c)
                
                # Crossover
                j_rand = rng.integers(0, dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, -5, 5)
                f_u = evaluate(u)
                
                # Selection
                if f_u <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = f_u
                    if f_u < global_best_fx:
                        global_best_fx = f_u
                        global_best_x = u.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            pop = new_pop
            fit = new_fit
            
            if no_improve > max_no_improve * pop_size:
                # Restart or break if stuck
                break
                
        # Phase 2: Local Refinement using Nelder-Mead
        # Use remaining budget
        remaining_budget = budget - calls_made
        if global_best_x is not None and remaining_budget > 10 and time.time() < deadline:
            try:
                res = minimize(
                    func, 
                    global_best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.success and res.fun < global_best_fx:
                    global_best_x = res.x
                    global_best_fx = res.fun
            except Exception:
                pass
            
            # Try L-BFGS-B as well if budget allows
            if calls_made < budget and time.time() < deadline:
                try:
                    res2 = minimize(
                        func, 
                        global_best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': (budget - calls_made) // 2, 'ftol': 1e-10}
                    )
                    if res2.success and res2.fun < global_best_fx:
                        global_best_x = res2.x
                        global_best_fx = res2.fun
                except Exception:
                    pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
