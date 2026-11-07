"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear model extraction with incremental vertex construction for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, kill at 290s to be safe
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

        # Phase 1: Incremental Vertex Construction
        # We determine the optimal vertex of the hypercube [-5, 5]^d
        # by probing each axis and moving to the optimal bound for that axis.
        
        # Start at origin
        x_cur = np.zeros(dim)
        f_cur = func(x_cur)
        calls_used += 1
        update_best(x_cur, f_cur)
        
        if time.time() > deadline:
            return best_x, best_f

        # Probe each dimension
        for k in range(dim):
            if time.time() > deadline:
                break
            
            # 1. Probe to determine the sign of the gradient component w_k
            x_probe = x_cur.copy()
            x_probe[k] += 5.0
            f_probe = func(x_probe)
            calls_used += 1
            update_best(x_probe, f_probe) # Probe might be better than current best? Unlikely if x_cur is optimized for prev dims, but possible if x_cur was 0 and probe is good.
            
            # Calculate slope w_k
            # f(x_cur + 5e_k) - f(x_cur) = 5 * w_k
            w_k = (f_probe - f_cur) / 5.0
            
            # 2. Move x_cur to the optimal bound for dimension k
            if w_k > 0:
                x_cur[k] = -5.0
            else:
                x_cur[k] = 5.0
                
            # 3. Evaluate the new point (which is better than x_cur for this dim)
            f_cur = func(x_cur)
            calls_used += 1
            update_best(x_cur, f_cur)
            
            # Check budget
            if calls_used >= budget:
                break

        # Phase 2: Polish
        # The function is linear, so the vertex is the global minimum.
        # However, we perform a small (1+1)-ES polish to handle any numerical
        # precision issues or slight non-linearities, though none are expected.
        
        if calls_used < budget and best_x is not None:
            x_cur = best_x.copy()
            f_cur = best_f
            sigma = 1e-3
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
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
