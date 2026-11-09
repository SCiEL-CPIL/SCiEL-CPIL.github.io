"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for linear slope function with finite difference gradient estimation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Phase 1: Initial exploration to find a reasonable starting point
        # Use a few random points to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        # Evaluate center and a few random points
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Try a few random points to find a better starting point
        candidates = [x0]
        f_candidates = [f0]
        
        for i in range(10):
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            candidates.append(x_rand)
            f_candidates.append(f_rand)
        
        # Pick the best starting point
        best_idx = np.argmin(f_candidates)
        x = candidates[best_idx].copy()
        fx = f_candidates[best_idx]
        
        # Phase 2: Estimate gradient using central differences
        # For a linear function, one gradient estimate is enough
        # Use a small step size for finite differences
        h = 0.1
        grad = np.zeros(dim)
        
        for i in range(dim):
            x_plus = x.copy()
            x_plus[i] = min(5.0, x_plus[i] + h)
            x_minus = x.copy()
            x_minus[i] = max(-5.0, x_minus[i] - h)
            
            # Adjust h if we hit the boundary
            actual_h_plus = x_plus[i] - x[i]
            actual_h_minus = x[i] - x_minus[i]
            
            if abs(actual_h_plus) < 1e-10:
                # At upper boundary, use forward difference
                f_plus = func(x_plus)
                grad[i] = (f_plus - fx) / max(h, 1e-10)
            elif abs(actual_h_minus) < 1e-10:
                # At lower boundary, use backward difference
                f_minus = func(x_minus)
                grad[i] = (fx - f_minus) / max(h, 1e-10)
            else:
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                grad[i] = (f_plus - f_minus) / (actual_h_plus + actual_h_minus)
        
        # Phase 3: Gradient descent with adaptive step size
        # For a linear function, the gradient is constant, so we can compute
        # the step to reach the boundary or use a fixed step
        
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            # Gradient is essentially zero, we're at the optimum
            return x, fx
        
        # Normalize gradient
        grad_unit = grad / grad_norm
        
        # We want to move in the direction that decreases f
        # Since f(x) = c·x + b, moving in -grad direction decreases f
        # The minimum will be at the boundary in the -grad direction
        
        # Compute how far we can go before hitting the boundary
        # x(t) = x - t * grad_unit
        # We need -5 <= x_i - t * grad_unit_i <= 5 for all i
        
        t_min = 0
        t_max = float('inf')
        
        for i in range(dim):
            if grad_unit[i] > 1e-10:
                # x_i - t * grad_unit_i >= -5  =>  t <= (x_i + 5) / grad_unit[i]
                t_max = min(t_max, (x[i] + 5) / grad_unit[i])
            elif grad_unit[i] < -1e-10:
                # x_i - t * grad_unit_i <= 5  =>  t <= (x[i] - 5) / grad_unit[i]
                # Since grad_unit[i] < 0, this is (x[i] - 5) / negative = (5 - x[i]) / |grad_unit[i]|
                t_max = min(t_max, (x[i] - 5) / grad_unit[i])
            # If grad_unit[i] ~ 0, no constraint from this dimension
        
        if t_max <= 0 or not np.isfinite(t_max):
            t_max = 1.0
        
        # Move towards the boundary in the gradient direction
        # For a linear function, the optimum is at the boundary
        t_step = t_max * 0.999  # Stay slightly inside
        
        x_new = x - t_step * grad_unit
        x_new = np.clip(x_new, -5, 5)
        f_new = func(x_new)
        
        if f_new < fx:
            x, fx = x_new, f_new
        
        # Phase 4: Fine-tuning with small steps
        # Use remaining budget for local refinement
        remaining_budget = budget - 1 - 10 - 2 * dim  # Approximate count
        remaining_budget = max(0, remaining_budget)
        
        # Re-estimate gradient at the new point for fine-tuning
        if remaining_budget > 0 and time.time() < deadline:
            h_fine = 0.01
            grad_fine = np.zeros(dim)
            
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] = min(5.0, x_plus[i] + h_fine)
                x_minus = x.copy()
                x_minus[i] = max(-5.0, x_minus[i] - h_fine)
                
                actual_h_plus = x_plus[i] - x[i]
                actual_h_minus = x[i] - x_minus[i]
                
                if abs(actual_h_plus) < 1e-10:
                    f_plus = func(x_plus)
                    grad_fine[i] = (f_plus - fx) / max(h_fine, 1e-10)
                elif abs(actual_h_minus) < 1e-10:
                    f_minus = func(x_minus)
                    grad_fine[i] = (fx - f_minus) / max(h_fine, 1e-10)
                else:
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    grad_fine[i] = (f_plus - f_minus) / (actual_h_plus + actual_h_minus)
            
            grad_fine_norm = np.linalg.norm(grad_fine)
            
            if grad_fine_norm > 1e-10:
                grad_fine_unit = grad_fine / grad_fine_norm
                
                # Small step in the gradient direction
                # Try several small steps to converge
                step_sizes = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
                
                for step in step_sizes:
                    if time.time() >= deadline:
                        break
                    
                    x_trial = x - step * grad_fine_unit
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    
                    if f_trial < fx:
                        x, fx = x_trial, f_trial
                    else:
                        break
        
        # Final check: make sure we haven't exceeded budget
        return x, fx
# EVOLVE-BLOCK-END
