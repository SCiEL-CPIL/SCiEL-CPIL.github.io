"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear model extraction with minimal polish for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Fit exact affine model f(x) = w^T x + b
        # Evaluate at origin and at 5*e_i for each axis to get maximum signal
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used += 1
        update_best(x0, f0)
        
        if time.time() > deadline:
            return best_x, best_f

        step = 5.0
        w = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = step
            fi = func(xi)
            calls_used += 1
            update_best(xi, fi)
            w[i] = (fi - f0) / step

        # Compute the optimal vertex: x[i] = -5 if w[i] > 0, +5 if w[i] < 0, 0 if w[i] == 0
        x_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 0:
                x_opt[i] = -5.0
            elif w[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the computed optimum
        f_opt_val = func(x_opt)
        calls_used += 1
        update_best(x_opt, f_opt_val)

        if time.time() > deadline:
            return best_x, best_f

        # Phase 2: Polish with (1+1)-ES using very small sigma
        # For a strictly linear function, the vertex is optimal
        # Use tiny steps to avoid leaving the vertex while catching numerical artifacts
        x_cur = x_opt.copy()
        f_cur = f_opt_val
        sigma = 1e-4
        rng = np.random.default_rng(42)
        
        remaining = budget - calls_used
        for _ in range(remaining):
            if time.time() > deadline:
                break
            
            y = x_cur + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls_used += 1
            update_best(y, fy)
            
            if fy < f_cur:
                x_cur = y
                f_cur = fy
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
