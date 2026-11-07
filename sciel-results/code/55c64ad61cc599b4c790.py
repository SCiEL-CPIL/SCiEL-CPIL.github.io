import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx
        
        # Phase 1: Deterministic Global Sampling
        # This ensures early progress by covering the domain systematically.
        # 1. Center
        # 2. 32 Vertices of [-5, 5]^5
        # 3. 10 Axis midpoints (±2.5 on each axis)
        
        points_to_eval = []
        
        # Center
        points_to_eval.append(np.zeros(n))
        
        # Vertices
        for i in range(2**n):
            vertex = []
            for j in range(n):
                if (i >> j) & 1:
                    vertex.append(5.0)
                else:
                    vertex.append(-5.0)
            points_to_eval.append(np.array(vertex))
            
        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)
        
        # Evaluate these points
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 10:
                break
            wrapped_func(p)
            
        # If we didn't find any points (shouldn't happen), fallback to random
        if best_x is None:
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, n)
            best_f = wrapped_func(best_x)

        # Phase 2: Local Refinement
        # Use the best point found in Phase 1 to start a local optimizer.
        # Nelder-Mead is robust for this type of function.
        if calls < budget and time.time() < deadline - 5:
            remaining_budget = budget - calls
            # We can use most of the remaining budget for refinement
            max_iter_final = min(remaining_budget, 400) # Cap to avoid infinite loops if it doesn't converge
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_final,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                # Note: wrapped_func updates best_f and best_x internally, 
                # so we don't strictly need to check res.fun, but it's good practice.
            except Exception:
                pass

        return best_x, best_f
