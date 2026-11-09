"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB-style f9 (Rosenbrock rotated).
    Strategy:
    1. Multi-start Nelder-Mead for global exploration.
    2. Powell's method for high-precision local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget or check_deadline():
                return np.inf
            # Clip to box [-5, 5]^n
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            return fx

        # Initial evaluation at origin
        x0 = np.zeros(n)
        f0 = do_call(x0)
        best_x = x0.copy()
        best_f = f0

        # Phase 1: Global Search via Multi-start Nelder-Mead
        # Allocate ~50% of budget to global search
        global_budget = int(budget * 0.5)
        
        # Generate diverse starting points
        # 1. Origin (already done)
        # 2. Grid points
        # 3. Random points
        start_points = []
        
        # Add origin
        start_points.append(np.zeros(n))
        
        # Add random points in [-4, 4]^n
        num_random_starts = 8
        for _ in range(num_random_starts):
            start_points.append(rng.uniform(-4, 4, n))
            
        # Add points near the origin but slightly perturbed
        for scale in [0.5, 1.0, 2.0]:
            start_points.append(scale * rng.standard_normal(n))

        # We have ~12 start points.
        # Budget per start: (global_budget - 1) / num_starts
        # We will run Nelder-Mead from each, stopping if budget runs out.
        
        starts_per_run = max(1, (global_budget - calls_used) // len(start_points))
        
        for i, sp in enumerate(start_points):
            if calls_used >= global_budget or check_deadline():
                break
            
            # Define objective for Nelder-Mead
            def objective_global(x):
                nonlocal calls_used
                if check_deadline() or calls_used >= global_budget:
                    raise ValueError("Budget exceeded in global phase")
                return do_call(x)

            try:
                # Nelder-Mead parameters
                # Initial simplex size: 0.05 * (1 + |x|) is standard, but for box [-5,5], 
                # a fixed small initial step is often better to avoid overshooting.
                # adaptive=True helps with scaling.
                res = minimize(
                    objective_global,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': starts_per_run,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                # If an exception occurs (e.g. budget limit), we just stop this run
                pass

        # Phase 2: Local Polish with Powell's Method
        # Powell's method is excellent for Rosenbrock as it optimizes along conjugate directions.
        remaining_budget = budget - calls_used
        
        if remaining_budget > 10 and not check_deadline():
            def objective_polish(x):
                nonlocal calls_used
                if check_deadline() or calls_used >= budget:
                    raise ValueError("Budget exceeded in polish phase")
                return do_call(x)

            try:
                # Powell's method doesn't have a simple 'maxiter' in all scipy versions 
                # that behaves like Nelder-Mead, but 'maxiter' limits the number of iterations.
                # We want to use as much of the remaining budget as possible.
                # Note: Powell's method can be slower per iteration than Nelder-Mead 
                # but converges more precisely.
                
                res = minimize(
                    objective_polish,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': remaining_budget,
                        'xtol': 1e-10,
                        'ftol': 1e-12
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        # Phase 3: Final Nelder-Mead Polish if time/budget remains
        # Sometimes Nelder-Mead can recover from Powell's potential stagnation
        remaining_budget = budget - calls_used
        if remaining_budget > 5 and not check_deadline():
            def objective_final(x):
                nonlocal calls_used
                if check_deadline() or calls_used >= budget:
                    raise ValueError("Budget exceeded in final phase")
                return do_call(x)

            try:
                res = minimize(
                    objective_final,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        # Ensure we have at least 1 call if something went wrong
        if calls_used == 0:
            do_call(np.zeros(n))

        return best_x, best_f
# EVOLVE-BLOCK-END
