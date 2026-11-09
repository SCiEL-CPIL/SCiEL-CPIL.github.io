import numpy as np
import time

class Optimizer:
    """
    Sequential Coordinate Descent Optimizer for BBOB f5 (Linear Slope).
    
    Strategy:
    1. Exploit the linearity of BBOB f5. The function is of the form f(x) = a^T x + b.
    2. The minimum over the box [-5, 5]^d is always at a vertex.
    3. Instead of estimating the full gradient vector via central differences (which uses 1 + 2*d calls 
       and evaluates points far from the boundary), we use a sequential 1D coordinate descent approach.
    4. For each coordinate i, we evaluate the function at the two bounds (-5, +5) while keeping 
       other coordinates at their current 'best' values (initially 0).
    5. We greedily fix the coordinate to the bound that gives the lower function value.
    6. This converges to the global vertex in exactly d + 1 calls (1 for center, 2*d for bounds).
    7. Since we evaluate points at the boundaries early, the early error e_i is much smaller, 
       leading to a higher AOCC score.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, use 280s to be safe against overhead
        self.deadline = self.start_time + 280.0
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        # 1. Wrapped function for strict tracking
        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                    
                return val
            except Exception:
                return float('inf')

        # 2. Phase 1: Sequential Coordinate Descent to find the optimal vertex
        # Start at the center
        x_center = np.zeros(self.dim)
        f_center = wrapped_func(x_center)
        
        if self.calls_made >= self.budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Current best point (initially center)
        x_curr = x_center.copy()
        f_curr = f_center
        
        # For each coordinate, evaluate both bounds and pick the better one
        for i in range(self.dim):
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            
            # Evaluate at -5 for coordinate i
            x_neg = x_curr.copy()
            x_neg[i] = -5.0
            f_neg = wrapped_func(x_neg)
            
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            
            # Evaluate at +5 for coordinate i
            x_pos = x_curr.copy()
            x_pos[i] = 5.0
            f_pos = wrapped_func(x_pos)
            
            # Greedily fix the coordinate to the better bound
            if f_neg < f_pos:
                x_curr[i] = -5.0
                f_curr = f_neg
            else:
                x_curr[i] = 5.0
                f_curr = f_pos
        
        # 3. Phase 2: Verification and Refinement
        # The sequential coordinate descent on a linear function should have found the global minimum.
        # However, due to potential numerical issues or if the function is not perfectly linear,
        # we can do a few refinement steps if budget remains.
        
        if self.budget - self.calls_made > 5:
            # Simple (1+1)-ES refinement from the found vertex
            x_curr = self.best_x.copy() if self.best_x is not None else x_curr
            f_curr = self.best_f
            sigma = 0.5
            rng = np.random.default_rng(42)
            
            for _ in range(10): # Max 10 refinement steps
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break
                
                y = np.clip(x_curr + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                    
                if sigma < 1e-4:
                    break

        return self.best_x, self.best_f
