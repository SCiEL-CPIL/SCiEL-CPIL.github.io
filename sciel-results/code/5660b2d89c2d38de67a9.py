"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Gradient Descent with central differences and polish for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(n)
        call_count = 0
        
        # Initial evaluation
        fx = func(x)
        call_count += 1
        
        # Adaptive parameters
        eps_base = 0.01  # Larger base epsilon for better gradient estimates
        step_size = 1.0  # Initial learning rate
        min_step = 1e-12
        max_step = 10.0
        
        # Phase 1: Main optimization with adaptive step
        # Reserve ~20 calls for polish phase
        polish_budget = min(20, budget // 10)
        main_budget = budget - polish_budget
        
        while call_count < main_budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Adaptive epsilon: larger when far from optimum, smaller when close
            # Use gradient magnitude to adapt
            # Start with a reasonable epsilon
            eps = eps_base
            
            # Estimate gradient using central differences (2*n calls)
            # But we need to check if we have enough budget
            if call_count + 2 * n + 1 > main_budget:
                # Not enough budget for full gradient, use forward differences
                if call_count + n + 1 > main_budget:
                    break
                # Forward difference
                grad = np.zeros(n)
                for i in range(n):
                    x_p = x.copy()
                    x_p[i] += eps
                    x_p = np.clip(x_p, -5.0, 5.0)
                    f_p = func(x_p)
                    call_count += 1
                    grad[i] = (f_p - fx) / eps
            else:
                # Central difference
                grad = np.zeros(n)
                for i in range(n):
                    x_p = x.copy()
                    x_m = x.copy()
                    x_p[i] += eps
                    x_m[i] -= eps
                    x_p = np.clip(x_p, -5.0, 5.0)
                    x_m = np.clip(x_m, -5.0, 5.0)
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2 * eps)
            
            # Check if we are at a minimum
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            # Adaptive step size: scale with gradient magnitude
            # For sphere f(x) = ||x - x*||^2 + c, grad = 2*(x - x*)
            # So step should be proportional to grad / (2*lambda) where lambda ~ 1
            # Use normalized gradient with adaptive step
            grad_dir = grad / (grad_norm + 1e-30)
            
            # Try a step in the negative gradient direction
            # Use a heuristic: step_size * grad_norm / some_scale
            # For sphere, optimal step is grad_norm / 2 (since f ~ ||grad||^2/4 + ...)
            proposed_step = step_size * grad_norm / 2.0
            proposed_step = np.clip(proposed_step, min_step, max_step)
            
            x_new = x - proposed_step * grad_dir
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # Evaluate new point
            f_new = func(x_new)
            call_count += 1
            
            if f_new < fx:
                # Accept step
                x = x_new
                fx = f_new
                # Increase step size slightly
                step_size = min(step_size * 1.1, max_step)
                # Reduce epsilon as we get closer
                eps_base = max(eps_base * 0.95, 1e-6)
            else:
                # Reject step, decrease step size
                step_size *= 0.7
                if step_size < min_step:
                    break
        
        # Phase 2: Polish phase with very small steps
        if call_count < budget:
            polish_eps = 1e-7
            polish_step = 1e-5
            for _ in range(min(polish_budget, budget - call_count)):
                if time.time() > deadline:
                    break
                
                # Fine gradient estimate
                grad = np.zeros(n)
                for i in range(n):
                    x_p = x.copy()
                    x_p[i] += polish_eps
                    x_p = np.clip(x_p, -5.0, 5.0)
                    f_p = func(x_p)
                    call_count += 1
                    if call_count >= budget:
                        break
                    grad[i] = (f_p - fx) / polish_eps
                if call_count >= budget:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-16:
                    break
                
                grad_dir = grad / (grad_norm + 1e-30)
                x_new = x - polish_step * grad_dir
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                call_count += 1
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    polish_step *= 0.9
                else:
                    polish_step *= 0.5
                    if polish_step < 1e-15:
                        break
        
        return x, fx
# EVOLVE-BLOCK-END
