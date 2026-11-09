import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_asa(x_start, T_start, alpha_cool, n_iter, rng_asa):
            nonlocal best_x, best_f, calls
            
            x_c = x_start.copy()
            f_c = try_call(x_c)
            if f_c is None:
                return
            
            T = T_start
            alpha_pull = 0.5  # Weight for pull towards best
            
            for _ in range(n_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate candidate
                # Direction towards best
                if best_x is not None and not np.allclose(x_c, best_x):
                    dir_best = best_x - x_c
                    dist = np.linalg.norm(dir_best)
                    if dist > 1e-10:
                        dir_best = dir_best / dist
                    else:
                        dir_best = np.zeros(dim)
                else:
                    dir_best = np.zeros(dim)
                
                # Random noise
                noise = rng_asa.standard_normal(dim)
                
                # Step size: proportional to temperature and distance to best
                # We scale the pull by T to allow larger jumps when T is high
                step = T * (alpha_pull * dir_best * 2.0 + noise)
                
                x_new = x_c + step
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = try_call(x_new)
                if f_new is None:
                    break
                
                # Acceptance criterion
                if f_new < f_c:
                    x_c = x_new
                    f_c = f_new
                else:
                    # Metropolis acceptance
                    delta_f = f_new - f_c
                    if T > 1e-10:
                        prob = np.exp(-delta_f / T)
                        if rng_asa.random() < prob:
                            x_c = x_new
                            f_c = f_new
                
                # Cool down
                T *= alpha_cool
                
                # If T becomes too small, we might be stuck. 
                # The restart logic outside will handle this.

        # Phase 1: Initialization
        # Evaluate origin and a few random points
        x0 = np.zeros(dim)
        f0 = try_call(x0)
        
        n_init = 10
        for _ in range(n_init):
            if calls >= budget:
                break
            xr = rng.uniform(-4, 4, size=dim)
            fr = try_call(xr)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = try_call(best_x)
            if best_f is None:
                return best_x, best_f

        # Phase 2: Multi-Start ASA
        # We will run ASA from the current best, and if it stagnates, restart from a random point.
        # Total budget for ASA phase: ~70%
        budget_asa = int(0.7 * budget)
        
        # We divide the ASA budget into chunks for multiple restarts
        n_restarts = 5
        budget_per_restart = budget_asa // n_restarts
        
        for i in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Determine starting point
            if i == 0:
                x_start = best_x.copy()
            else:
                # Restart from a random point or a point near the best
                # Using a random point helps escape deep local minima
                x_start = rng.uniform(-4, 4, size=dim)
            
            # Temperature starts high to allow global exploration, then cools
            T_start = 2.0
            alpha_cool = 0.995  # Slow cooling to maintain exploration
            
            run_asa(x_start, T_start, alpha_cool, budget_per_restart, rng)
            
            # Check for stagnation or time
            if time.time() > deadline - 10:
                break

        # Phase 3: Polish with Nelder-Mead
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize
                
                remaining = budget - calls
                if remaining > 10:
                    # Use Nelder-Mead for local refinement
                    result = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': min(remaining, 100),
                                 'xatol': 1e-8,
                                 'fatol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_x = np.clip(result.x, -5.0, 5.0)
                        best_f = result.fun
            except Exception:
                pass
        
        return best_x, best_f
