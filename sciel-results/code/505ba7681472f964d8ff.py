import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Parameters for Differential Evolution
        npop = 10  # Population size
        F = 0.8    # Mutation factor
        CR = 0.9   # Crossover rate
        
        # Allocate budget: 60% for DE, 40% for Local Polish
        de_budget = int(budget * 0.6)
        polish_budget = budget - de_budget

        # Initialize population
        pop = rng.uniform(-5, 5, (npop, dim))
        fit = np.array([func(x) for x in pop])
        calls = npop

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_fx = fit[best_idx]

        # Differential Evolution Loop
        # We have de_budget - npop calls left for DE iterations
        # Each iteration uses npop calls
        max_de_iters = (de_budget - calls) // npop

        for gen in range(max_de_iters):
            if time.time() >= deadline:
                break
            if calls >= de_budget:
                break

            for i in range(npop):
                if calls >= de_budget:
                    break
                if time.time() >= deadline:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(npop, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                ft = func(trial)
                calls += 1
                
                # Selection
                if ft < fit[i]:
                    pop[i] = trial
                    fit[i] = ft
                    if ft < best_fx:
                        best_x = trial.copy()
                        best_fx = ft

        # Local Polish Phase
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            # Use Powell's method for local refinement
            # Powell is effective for anisotropic functions
            nm_calls = [0]
            
            def polish_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(z)

            try:
                # Powell's method
                res = minimize(
                    polish_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': remaining,
                        'xtol': 1e-12,
                        'ftol': 1e-14
                    }
                )
                if res.fun < best_fx:
                    best_x = res.x
                    best_fx = res.fun
            except Exception:
                # Fallback to Nelder-Mead if Powell fails or times out
                nm_calls[0] = 0
                try:
                    res = minimize(
                        polish_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_fx:
                        best_x = res.x
                        best_fx = res.fun
                except Exception:
                    pass

        return best_x, best_fx
