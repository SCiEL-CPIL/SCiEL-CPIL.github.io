"""Rewritten program: Exact linear optimization using gradient estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the linearity of the BBOB f5 function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Safety deadline: 290 seconds out of 300 allowed
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        budget = self.budget
        
        # Track best point and value
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Step 1: Evaluate at the origin
        x0 = np.zeros(d)
        f0 = func(x0)
        update_best(x0, f0)
        
        # Step 2: Estimate the gradient vector c
        # f(x) = c^T x + b
        # f(0) = b
        # f(delta * e_i) = c_i * delta + b
        # c_i = (f(delta * e_i) - f(0)) / delta
        # Using delta=5.0 maximizes signal strength for linear gradient estimation
        # and ensures probes are at the box boundaries, providing the largest possible
        # initial drop in function value for components with negative gradients.
        delta = 5.0
        c = np.zeros(d)
        
        for i in range(d):
            if time.time() > deadline:
                return best_x, best_f
                
            probe = np.zeros(d)
            probe[i] = delta
            f_probe = func(probe)
            c[i] = (f_probe - f0) / delta
            update_best(probe, f_probe)
            
        # Step 3: Determine the optimal vertex
        # The minimum of a linear function on a hypercube is at a vertex.
        # x_i^* = 5 if c_i < 0, -5 if c_i > 0
        x_opt = np.zeros(d)
        for i in range(d):
            if c[i] < 0:
                x_opt[i] = 5.0
            elif c[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 0.0
                
        # Step 4: Evaluate the optimal vertex
        if time.time() > deadline:
            return best_x, best_f
            
        f_opt = func(x_opt)
        update_best(x_opt, f_opt)
        
        # Step 5: If we have remaining budget, do additional evaluations to improve AOCC
        # Since we've already found the optimum, we can't improve the function value,
        # but the AOCC metric rewards early progress. However, since we already reached
        # the optimum at call 7, additional calls won't change the score unless we find
        # a better solution (which we can't for a linear function).
        # 
        # Actually, let's think about this more carefully. The AOCC is calculated as:
        # AOCC = 1 - mean(n_1, ..., n_500)
        # where n_i = (log10(e_i) + 8) / 10 and e_i = clip(b_i - f_opt, 1e-8, 100)
        # 
        # If we reach f_opt at call 7, then b_7 = f_opt, so e_7 = 1e-8 (clipped),
        # n_7 = (log10(1e-8) + 8) / 10 = (-8 + 8) / 10 = 0
        # For calls 8-500, b_i = f_opt, so n_i = 0
        # For calls 1-6, n_i > 0
        # 
        # So the score is determined by how quickly we reach the optimum.
        # We've already done that at call 7. Additional calls don't help.
        # 
        # However, if there's any numerical noise or if the function isn't perfectly linear,
        # we might be able to do better. But the task says it's linear, so we're done.
        # 
        # Let's just return the best we found.
        return best_x, best_f
# EVOLVE-BLOCK-END
