"""Rewritten program using Large-Step Gradient Estimation for Linear Slope."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer exploiting the linear structure of f5 (Linear slope).
    Uses large steps for gradient estimation to improve early progress metrics (AOCC).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # 1. Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # 2. Estimate Gradient Signs using Large Steps
        # Use step h=1.0 to get significant improvement in early calls.
        # This helps reduce n_i for i=2..6.
        h = 1.0
        grad_signs = np.zeros(dim)
        
        # Check if we have enough budget
        if calls + dim < budget:
            for i in range(dim):
                if time.time() > deadline:
                    break
                x_test = x.copy()
                x_test[i] += h
                fy_test = func(x_test)
                calls += 1
                
                # Update best if this point is better
                if fy_test < best_fx:
                    best_fx = fy_test
                    best_x = x_test.copy()
                
                # Estimate gradient component
                # f(x+h) - f(x) approx h * g_i
                # We only need the sign.
                diff = fy_test - fx
                grad_signs[i] = diff / h # Magnitude is useful too, but sign is key for corner
                
                # Early termination if time is critical
                if time.time() > deadline:
                    break
        else:
            # Fallback if budget is too small (unlikely)
            pass

        # 3. Predict Minimum Corner
        # If grad[i] > 0, function increases with x[i], so min is at -5.
        # If grad[i] < 0, function decreases with x[i], so min is at 5.
        # If grad[i] == 0, direction doesn't matter, pick 0 or arbitrary.
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad_signs[i] > 0:
                x_pred[i] = -5.0
            elif grad_signs[i] < 0:
                x_pred[i] = 5.0
            else:
                x_pred[i] = 0.0 # Flat, stay at 0? Or random? 0 is safe.

        # Evaluate the predicted corner
        if calls < budget:
            fx_pred = func(x_pred)
            calls += 1
            if fx_pred < best_fx:
                best_fx = fx_pred
                best_x = x_pred.copy()
        
        # 4. Refinement Phase (Safety Net)
        # If the linear assumption was wrong or noise caused a bad prediction,
        # use (1+1)-ES to recover.
        # Start from best_x.
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        
        success_counter = 0
        total_steps = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                sigma *= 1.2
                success_counter += 1
            else:
                sigma *= 0.9
                success_counter = 0
            
            total_steps += 1
            
            # Simple restart logic if stuck
            if total_steps > 50 and success_counter == 0:
                x = best_x.copy()
                fx = best_fx
                sigma = 1.0
                total_steps = 0
                success_counter = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
