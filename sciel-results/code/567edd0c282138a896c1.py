"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gradient Descent with adaptive step size and polish phase for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        call_count = 1
        
        best_x = x.copy()
        best_fx = fx
        
        # Gradient estimation parameters - larger eps for numerical stability
        eps = 1e-4
        
        # Adaptive step size
        alpha = 1.0
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a central difference gradient
            if call_count + 2 * dim > budget:
                if call_count + dim <= budget:
                     # Forward difference fallback
                     grad = np.zeros(dim)
                     for i in range(dim):
                         x_p = x.copy()
                         x_p[i] += eps
                         f_p = func(x_p)
                         call_count += 1
                         grad[i] = (f_p - fx) / eps
                else:
                    break
            else:
                # Central difference
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x.copy()
                    x_p[i] += eps
                    x_m = x.copy()
                    x_m[i] -= eps
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2 * eps)
            
            # Check convergence
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq < 1e-20:
                break
            
            # Backtracking line search
            accepted = False
            current_alpha = alpha
            
            # Limit inner retries to avoid wasting budget on one step
            for _ in range(10): 
                x_new = x - current_alpha * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                call_count += 1
                
                if f_new < fx:
                    # Accept
                    accepted = True
                    
                    # Update best
                    x = x_new
                    fx = f_new
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    
                    # Increase alpha if we took a full step
                    alpha = min(current_alpha * 1.1, 10.0)
                    break
                else:
                    # Reject, shrink step
                    current_alpha *= 0.5
                    if current_alpha < 1e-10:
                        break

            if not accepted:
                # If we couldn't find a step even with very small alpha, 
                # we might be at a minimum or in a flat region.
                break

        # Polish phase: use smaller eps and more careful steps
        if call_count < budget and time.time() < deadline:
            eps_polish = 1e-6
            alpha_polish = 0.1
            for _ in range(20):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Central difference with smaller eps
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x.copy()
                    x_p[i] += eps_polish
                    x_m = x.copy()
                    x_m[i] -= eps_polish
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2 * eps_polish)
                
                grad_norm_sq = np.dot(grad, grad)
                if grad_norm_sq < 1e-24:
                    break
                
                # Small step
                x_new = x - alpha_polish * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                call_count += 1
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    alpha_polish *= 0.5
                    if alpha_polish < 1e-12:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
