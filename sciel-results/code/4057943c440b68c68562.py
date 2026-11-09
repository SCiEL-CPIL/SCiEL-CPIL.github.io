# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    For a linear function f(x) = c + v^T x, the minimum over [-5,5]^d
    is at a vertex. We determine the optimal vertex by finding the sign
    of each gradient component using d+1 evaluations (simplex from origin),
    then evaluate that vertex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Step 2: Evaluate along each positive axis at distance 5
        # f(5*e_i) = f0 + 5 * v_i => v_i = (f(5*e_i) - f0) / 5
        v = np.zeros(d)
        v_known = np.zeros(d, dtype=bool)
        
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            v[i] = (fi - f0) / 5.0
            v_known[i] = True
            
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
            
            # After each axis eval, compute partial optimal vertex
            # For known components, use optimal sign; for unknown, use 0
            x_partial = np.zeros(d)
            for j in range(d):
                if v_known[j]:
                    if v[j] < 0:
                        x_partial[j] = 5.0
                    else:
                        x_partial[j] = -5.0
                else:
                    x_partial[j] = 0.0
            
            if calls_made < self.budget and time.time() < deadline:
                fp = func(x_partial)
                calls_made += 1
                if fp < f_best:
                    f_best = fp
                    x_best = x_partial.copy()
        
        # Step 3: If we have all components, the full vertex is optimal
        if np.all(v_known):
            x_vertex = np.zeros(d)
            for i in range(d):
                if v[i] < 0:
                    x_vertex[i] = 5.0
                else:
                    x_vertex[i] = -5.0
            
            if calls_made < self.budget and time.time() < deadline:
                fv = func(x_vertex)
                calls_made += 1
                if fv < f_best:
                    f_best = fv
                    x_best = x_vertex.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
