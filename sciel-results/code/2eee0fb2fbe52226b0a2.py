"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate gradient using centered differences at the origin.
    2. Jump to the vertex of the box that minimizes the estimated linear model.
    3. Refine using (1+1)-ES with the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin (330s limit)
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        budget_used = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        # Phase 1: Gradient Estimation
        # Use centered differences around 0.
        # f(0)
        x0 = np.zeros(dim)
        f0 = func(x0)
        budget_used += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # Step size for gradient estimation
        delta = 1.0
        grad = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = np.clip(x0 + delta * e_i, -5, 5)
            x_minus = np.clip(x0 - delta * e_i, -5, 5)
            
            f_plus = func(x_plus)
            budget_used += 1
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
                
            f_minus = func(x_minus)
            budget_used += 1
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
                
            grad[i] = (f_plus - f_minus) / (2 * delta)

        # Phase 2: Informed Jump
        # The linear model is f(x) ~ f0 + grad^T x
        # To minimize, we want x_i = -5 if grad[i] > 0, else 5.
        # If grad[i] is 0, x_i can be anything, let's say 0.
        x_jump = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 1e-6:
                x_jump[i] = -5.0
            elif grad[i] < -1e-6:
                x_jump[i] = 5.0
            else:
                x_jump[i] = 0.0

        if budget_used < budget:
            f_jump = func(x_jump)
            budget_used += 1
            if f_jump < best_f:
                best_f = f_jump
                best_x = x_jump.copy()

        # Phase 3: Local Refinement (1+1)-ES
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Moderate start, will adapt
        
        # If we are at a boundary, we might want to be careful.
        # But (1+1)-ES with clipping handles boundaries well.
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.85
                
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Optional: Early stopping if sigma becomes very small and no improvement
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
