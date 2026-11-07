import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
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
                best_x = x.copy()
                best_f = f

        def make_wrapper(fallback_val):
            def wrapper(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return fallback_val
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            return wrapper

        # Phase 1: Initial Exploration
        # Rosenbrock in 5D is narrow. Random uniform in [-5,5] is bad.
        # But we have to start somewhere. Let's try a few points.
        # Maybe bias towards the origin or known good regions? 
        # The problem statement says "hidden seeded shift, rotation and offset". 
        # So we can't rely on (1,1,1,1,1). We must search blindly.
        
        n_explore = 15
        candidates = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            # Sample uniformly in the box
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            candidates.append((x.copy(), fx))

        # Sort candidates by function value
        candidates.sort(key=lambda c: c[1])
        
        # Phase 2: Local Optimization with SLSQP
        # SLSQP is efficient for smooth functions with bounds.
        # We will run SLSQP on the top K candidates.
        # Each SLSQP run might take ~10-50 calls.
        # Budget is 500. We used 15. Remaining ~485.
        # We can afford ~10-15 runs.
        
        max_runs = 12
        for i in range(min(max_runs, len(candidates))):
            if not time_ok() or calls_made >= budget:
                break
            
            sx, sf = candidates[i]
            
            # If this candidate is already worse than best_f by a lot, skip?
            # No, SLSQP might escape a bad local minimum if the landscape is tricky, 
            # but for Rosenbrock, if you're far, you're stuck.
            # However, the "hidden shift" means the valley is somewhere.
            # If sf is very high, it's unlikely to be in the valley.
            # But let's try the top ones.
            
            wrapper = make_wrapper(best_f)
            
            try:
                # SLSQP parameters
                # ftol and gtol control convergence
                # maxfun limits evaluations
                result = minimize(
                    wrapper,
                    sx,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': 30 # Limit per run to ensure we try multiple starts
                    }
                )
            except Exception:
                pass
            
            # After SLSQP, if we found a better point, add it to candidates for potential re-refinement?
            # Or just update best.
            
        # Phase 3: Refinement of the Global Best
        # If we have budget left, refine the best point found so far.
        if best_x is not None and calls_made < budget and time_ok():
             # Run SLSQP again from the best point with tighter tolerances
             wrapper = make_wrapper(best_f)
             try:
                 result = minimize(
                     wrapper,
                     best_x.copy(),
                     method='SLSQP',
                     bounds=[(-5, 5)] * dim,
                     options={
                         'maxiter': 200,
                         'ftol': 1e-14,
                         'gtol': 1e-10,
                         'maxfun': 50
                     }
                 )
             except Exception:
                 pass

        # Phase 4: Final Polish with L-BFGS-B or Nelder-Mead?
        # SLSQP gives a good local min. L-BFGS-B might polish it further if it's stuck.
        if best_x is not None and calls_made < budget and time_ok():
             wrapper = make_wrapper(best_f)
             try:
                 result = minimize(
                     wrapper,
                     best_x.copy(),
                     method='L-BFGS-B',
                     bounds=[(-5, 5)] * dim,
                     options={
                         'maxiter': 100,
                         'ftol': 1e-15,
                         'gtol': 1e-12,
                         'maxfun': 20
                     }
                 )
             except Exception:
                 pass

        return best_x, best_f
