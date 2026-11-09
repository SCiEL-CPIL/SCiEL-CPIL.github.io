import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        # 1. Initialization with Halton sequence for better spread
        # Halton sequence in dim dimensions
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        for i in range(1, 11): # 10 initial points
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + self.dim)])
            # Map [0,1] to [-5,5]
            p = p * 10.0 - 5.0
            points.append(p)
        
        # Evaluate initial points
        f_values = []
        for p in points:
            f_val = func(p)
            f_values.append(f_val)
            
        # Find best
        best_idx = np.argmin(f_values)
        x_best = points[best_idx]
        f_best = f_values[best_idx]
        
        # 2. Construct adaptive initial simplex
        # Standard Nelder-Mead simplex in d-dim has d+1 vertices.
        # We want to elongate it along the direction of the second best point if possible.
        # Let's just use a standard simplex but scaled.
        # The default NM in scipy creates a simplex with side length 0.05 * (x + 0.05) roughly.
        # We can provide an initial_simplex.
        
        # Let's try to create a simplex that is elongated in the direction of the gradient estimate
        # or just use the direction between best and 2nd best.
        
        # If we have a 2nd best, use it to define a direction
        sorted_indices = np.argsort(f_values)
        x_2nd = points[sorted_indices[1]]
        
        # Direction from 2nd to best
        dir_vec = x_best - x_2nd
        norm_dir = np.linalg.norm(dir_vec)
        if norm_dir < 1e-6:
            dir_vec = np.array([1.0, 0, 0, 0, 0])[:self.dim]
        else:
            dir_vec = dir_vec / norm_dir

        # Create initial simplex
        # Vertex 0 is x_best
        # Other vertices are x_best + step * basis_vector
        # We make the step larger in the direction of dir_vec
        
        step_size = 1.0 # Initial step size
        
        simplex = [x_best.copy()]
        # Create basis vectors
        for i in range(self.dim):
            v = np.zeros(self.dim)
            v[i] = step_size
            # If this dimension aligns with dir_vec, make it bigger?
            # Actually, let's just use a standard simplex but maybe larger.
            simplex.append(x_best + v)
            
        # Alternatively, just let scipy handle it, but with a larger initial step?
        # scipy.optimize.minimize has 'initial_simplex' option.
        
        # Let's use the standard simplex but ensure it's not too small.
        # Default initial simplex in scipy NM is based on the input point.
        # If x is 0, it uses 0.05. If x is large, it scales.
        
        # Let's just run minimize with maxiter set to use most of the budget.
        # Budget used so far: 10
        # Remaining: 490
        # NM iterations: each iteration is roughly 1 func call + some overhead.
        # Let's allow 450 iterations.
        
        best_x = x_best
        best_f = f_best
        
        # Use a wrapper to track best and handle deadline
        eval_count = 10
        
        def wrapped_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        try:
            res = minimize(wrapped_func, x_best, method='Nelder-Mead', 
                           options={'maxiter': 450, 'xatol': 1e-8, 'fatol': 1e-8})
        except Exception as e:
            # If it fails, return best found so far
            pass

        return best_x, best_f
