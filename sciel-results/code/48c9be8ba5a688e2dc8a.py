"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid optimizer: Random exploration + L-BFGS-B local refinement + Differential Evolution + SLSQP high-precision polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def wrapped_func(x):
            """Wrapper that counts calls and tracks best."""
            nonlocal calls_made
            if not time_ok() or calls_made >= budget:
                return 1e10
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        # Phase 1: Random Exploration to find diverse seeds
        n_explore = 15
        seeds = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)
            seeds.append((x.copy(), fx))
            
            if calls_made >= budget:
                break

        if not seeds:
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return best_x, best_f

        # Sort seeds by fitness to pick the best few for local optimization
        seeds.sort(key=lambda s: s[1])
        
        # Phase 2: L-BFGS-B Local Optimization from best seeds
        # L-BFGS-B is efficient for smooth functions like Rosenbrock.
        n_local_runs = 4
        # Allocate roughly half the remaining budget for local optimization, split among runs
        calls_per_local = max(10, (budget - calls_made) // (2 * n_local_runs))
        
        for i in range(min(n_local_runs, len(seeds))):
            if not time_ok() or calls_made >= budget:
                break
            
            start_x, start_f = seeds[i]
            
            try:
                res = minimize(wrapped_func, start_x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 80, 'maxfun': calls_per_local,
                                        'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun
            except Exception:
                pass

        # Phase 3: Differential Evolution for global exploration if budget remains
        # DE helps escape local minima and find better basins for the narrow Rosenbrock valley.
        if best_x is not None and calls_made < budget - 20 and time_ok():
            remaining = budget - calls_made
            if remaining > 50:
                try:
                    # Use Differential Evolution with small population to explore around best
                    pop_size = min(10, max(5, dim + 4))
                    max_iter = max(5, remaining // (2 * pop_size))
                    
                    def de_func(x):
                        nonlocal calls_made
                        if not time_ok() or calls_made >= budget:
                            return 1e10
                        fx = func(x)
                        calls_made += 1
                        try_update(x, fx)
                        return fx
                    
                    # Start DE from best point with moderate sigma
                    de_res = differential_evolution(
                        de_func, 
                        bounds=[(-5, 5)] * dim,
                        x0=best_x,
                        popsize=pop_size,
                        maxiter=max_iter,
                        tol=1e-10,
                        seed=42,
                        polish=False,
                        init='sobol'
                    )
                    if de_res.fun < best_f:
                        best_x = de_res.x.copy()
                        best_f = de_res.fun
                except Exception:
                    pass

        # Phase 4: SLSQP High-Precision Polishing
        # SLSQP uses quadratic convergence near the optimum, ideal for final polishing.
        if best_x is not None and calls_made < budget and time_ok():
            try:
                slsqp_res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if slsqp_res.fun < best_f:
                    best_x = slsqp_res.x.copy()
                    best_f = slsqp_res.fun
            except Exception:
                pass

        # Phase 5: Final (1+1)-ES Polishing
        # Use remaining budget for fine-grained search around the best point found.
        if best_x is not None and calls_made < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            stall = 0
            max_stall = 30
            
            while calls_made < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y.copy(), fy
                    sigma = min(sigma * 1.05, 0.5)
                    stall = 0
                else:
                    sigma = max(sigma * 0.95, 1e-8)
                    stall += 1
                    if stall > max_stall:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
