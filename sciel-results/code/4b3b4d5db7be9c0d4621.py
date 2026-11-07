"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    f5 is a linear function: f(x) = f0 + c^T (R(x - shift))
    where R is a rotation, shift is a translation, c is a fixed vector.
    
    The minimum on [-5,5]^5 is at a corner of the box. We estimate the
    gradient direction via finite differences, then jump to the optimal corner.
    
    Strategy:
    1. Evaluate at origin to get f0.
    2. Estimate gradient via central differences (2*dim evaluations).
    3. Jump to the corner that minimizes the linear function.
    4. Verify by checking the 5 immediate neighbors of the corner.
    5. Use remaining budget for a simple local search / verification.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

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

        # Phase 1: Estimate gradient using central differences from origin
        # Use a step large enough to avoid numerical noise but small enough
        # to stay within the box for the origin-based evaluation
        h = 0.5
        x0 = np.zeros(dim)
        f0 = eval_point(x0)

        # Estimate gradient components
        grad = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h
            xm[i] -= h
            
            fp = eval_point(np.clip(xp, -5, 5))
            fm = eval_point(np.clip(xm, -5, 5))
            grad[i] = (fp - fm) / (2 * h)
        
        # Determine the corner that minimizes the linear function
        # For linear f(x) ≈ f0 + grad^T (x - x0), min is at
        # x_i = -5 if grad[i] > 0, else +5
        corner = np.where(grad > 0, -5.0, 5.0)
        
        # Evaluate the corner
        f_corner = eval_point(np.clip(corner, -5, 5))

        # Phase 2: Neighbor Verification
        # If the gradient estimation was wrong for one dimension, the true min is a neighbor.
        # Check all 5 neighbors (flip one bit)
        if calls_made < budget and time.time() < deadline:
            for i in range(dim):
                if time.time() > deadline:
                    break
                xn = corner.copy()
                xn[i] = -xn[i]
                fn = eval_point(xn)
        
        # Phase 3: Local Polish
        # Use remaining budget for a precise local search.
        # Since the function is linear, we expect to be at the min already.
        # But if there's any non-linearity or if we missed the corner, this helps.
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            sigma = 1.0
            
            # Run a limited number of iterations to avoid wasting budget
            max_iters = min(budget - calls_made, 100)
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                if sigma < 1e-10:
                    break
                
                y = x + sigma * np.random.randn(dim)
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                
                if fy < best_f:
                    x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
