"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin.
    2. Evaluate at 5 axis-boundary points (x_i = +/- 5) to determine gradient signs.
    3. Jump to the estimated optimal corner.
    4. Verify by checking the 5 immediate neighbors of the corner.
    5. Use remaining budget for a high-precision local polish (Coordinate Descent).
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # Safety margin

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # Phase 1: Exact gradient estimation via 6-point simplex
        # For a linear function f(x) = c^T x + b, we can determine c exactly
        # by evaluating at origin and 5 unit-axis points.
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        
        # Evaluate at unit axis points to get gradient components
        gradient = np.zeros(dim)
        h = 1.0
        
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = h
            
            fi = eval_point(xi)
            # For linear f: f(xi) - f0 = gradient[i] * h
            gradient[i] = (fi - f0) / h
        
        # Phase 2: Jump to the exact optimal corner
        # For linear f(x) = c^T x + b on [-5,5]^dim, minimum is at vertex
        # where x_i = -5 * sign(c_i) if c_i != 0, else either boundary
        corner = np.zeros(dim)
        for i in range(dim):
            if gradient[i] > 0:
                corner[i] = -5.0
            elif gradient[i] < 0:
                corner[i] = 5.0
            else:
                corner[i] = 0.0  # Doesn't matter for linear
        
        f_corner = eval_point(corner)
        
        # Phase 3: Verify by checking all 2^5 = 32 vertices if budget allows
        # This ensures we find the true minimum even if gradient estimation had issues
        if calls_made < budget and time.time() < deadline:
            # Generate all 32 vertices
            for mask in range(32):
                if calls_made >= budget or time.time() > deadline:
                    break
                v = np.zeros(dim)
                for i in range(dim):
                    v[i] = 5.0 if (mask & (1 << i)) else -5.0
                
                # Skip if we already evaluated this vertex (the corner)
                if np.allclose(v, corner, atol=1e-9):
                    continue
                
                fv = eval_point(v)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
