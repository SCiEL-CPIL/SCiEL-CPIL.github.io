"""Multi-restart Coordinate Descent with Nelder-Mead polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart Coordinate Descent with Nelder-Mead polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def coordinate_descent(x0, f0, max_calls):
            """
            Adaptive Coordinate Descent.
            Tries to improve one dimension at a time.
            """
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Initial step sizes
            steps = np.ones(n) * 0.5
            min_step = 1e-6

            # We track local calls to ensure we don't exceed max_calls for this phase
            local_calls = 0

            while local_calls < max_calls and time_ok() and calls < budget:
                improved = False
                for d in range(n):
                    if local_calls >= max_calls or not time_ok() or calls >= budget:
                        break
                    
                    # Try positive step
                    y = x.copy()
                    y[d] += steps[d]
                    y = np.clip(y, -5, 5)
                    fy = F(y)
                    local_calls += 1
                    
                    if fy < fx:
                        x = y.copy()
                        fx = fy
                        improved = True
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        # Try negative step if positive failed
                        y = x.copy()
                        y[d] -= steps[d]
                        y = np.clip(y, -5, 5)
                        fy = F(y)
                        local_calls += 1
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved = True
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                        else:
                            # Both directions failed, reduce step size for this dimension
                            steps[d] *= 0.5
                            if steps[d] < min_step:
                                steps[d] = min_step

                if not improved:
                    # If no dimension improved, try a small random kick to escape local minima
                    if steps.min() > min_step:
                        x = x + rng.standard_normal(n) * 0.1
                        x = np.clip(x, -5, 5)
                        fx = F(x)
                        local_calls += 1
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        break

            return x, fx

        def nelder_mead_polish(x0, f0):
            """
            Use SciPy's Nelder-Mead for final high-precision polish.
            """
            nonlocal best_x, best_f, calls
            
            try:
                # NM can be expensive. We limit maxiter to keep it bounded.
                # In dim=5, NM is quite efficient.
                res = minimize(
                    F, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            return best_x, best_f

        # Phase 1: Multi-restart Coordinate Descent
        n_restarts = 6
        # Allocate budget:
        # 500 calls total.
        # 6 restarts.
        # Each restart gets ~50 calls for CD. 6*50 = 300.
        # Remaining 200 for NM? That's a lot for NM.
        # Maybe 100 calls per CD restart? 6*100 = 600 > 500.
        # Let's do 5 restarts. 5 * 80 = 400 calls.
        # 100 calls left for NM.
        
        n_restarts = 5
        calls_per_cd = 80
        calls_reserved_for_nm = 100

        # Generate diverse starting points
        start_points = []
        start_points.append(np.zeros(dim))
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Shuffle to randomize which one gets "first" if budget runs out
        rng.shuffle(start_points)

        # Evaluate start points first to have a baseline
        # This uses n_restarts calls.
        # We need to be careful: if we evaluate sp, then run CD from sp, 
        # CD will start with f0 = F(sp). But we already paid for F(sp).
        # So we should pass the evaluated value to CD.
        
        # However, CD function above expects f0.
        # Let's evaluate all start points first.
        
        # But if budget is tight, we might not evaluate all.
        # Let's just run CD from each, and CD will evaluate the start point 
        # if it's not already the best? No, CD assumes f0 is known.
        
        # Better approach:
        # 1. Evaluate sp.
        # 2. Run CD from sp with f_sp.
        
        for sp in start_points:
            if calls >= budget - calls_reserved_for_nm or not time_ok():
                break
            
            # Evaluate start point
            f_sp = F(sp)
            if f_sp < best_f:
                best_f, best_x = f_sp, sp.copy()
            
            # Run CD from this point
            # We reserve some budget for NM, so we limit CD calls
            # Available for CD: budget - calls - calls_reserved_for_nm
            available = budget - calls - calls_reserved_for_nm
            if available < 10:
                break
                
            # Limit CD to a reasonable amount per restart
            # If we have 5 restarts and 400 budget, 80 each.
            # But we also used 1 call for sp.
            # So 79 left for CD logic?
            # Let's just use max_calls = min(calls_per_cd, available)
            
            coordinate_descent(sp, f_sp, min(calls_per_cd, available))

        # Phase 2: Nelder-Mead Polish
        if calls < budget and time_ok():
            nelder_mead_polish(best_x.copy(), best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
