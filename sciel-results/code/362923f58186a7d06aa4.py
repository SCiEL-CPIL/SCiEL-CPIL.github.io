"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic model-based optimizer for the BBOB Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 295.0  # Safety margin before 330s kill

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Construct a quadratic model from 2n+1 points
        # For Sphere: f(x) = ||x - z||^2 + offset = x^T I x - 2 z^T x + ||z||^2 + offset
        # We can estimate z (the minimizer) directly from the quadratic model.
        
        # Use a symmetric design around the center to build a quadratic model
        # Points: center, center ± delta * e_i
        # This gives us 2n+1 points to estimate the gradient and Hessian
        
        center = np.zeros(dim)
        delta = 2.0  # Step size for quadratic model construction
        
        # Evaluate center
        f_center = call_func(center)
        
        # Evaluate center + delta * e_i and center - delta * e_i
        # This allows us to estimate the gradient (first derivatives)
        # and the second derivatives (diagonal of Hessian)
        
        grad = np.zeros(dim)
        hess_diag = np.zeros(dim)
        
        for i in range(dim):
            if call_count >= budget or time.time() - start_time > timeout:
                break
                
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = center + delta * e_i
            x_minus = center - delta * e_i
            
            f_plus = call_func(x_plus)
            f_minus = call_func(x_minus)
            
            # Central difference for gradient
            grad[i] = (f_plus - f_minus) / (2.0 * delta)
            
            # Second difference for Hessian diagonal
            # f(x+h) - 2f(x) + f(x-h) ≈ h^2 * f''(x)
            # For sphere: f''(x_i) = 2 (since f = sum (x_i - z_i)^2 + offset)
            hess_diag[i] = (f_plus - 2.0 * f_center + f_minus) / (delta ** 2)
        
        # For Sphere function: f(x) = sum (x_i - z_i)^2 + offset
        # grad_i = 2 * (x_i - z_i)
        # So z_i = x_i - grad_i / 2
        # At center (zeros): z_i = 0 - grad_i / 2 = -grad_i / 2
        
        # Estimate the minimizer z
        z_est = -grad / 2.0
        
        # Clip to valid range
        z_est = np.clip(z_est, -4.5, 4.5)
        
        # Evaluate the estimated minimizer
        if call_count < budget and time.time() - start_time <= timeout:
            f_z = call_func(z_est)
        
        # Phase 2: Local refinement using the quadratic model
        # Use the estimated z as the starting point for a local search
        # We can do a simple gradient descent or use the quadratic model to take steps
        
        # For the Sphere function, the gradient is 2*(x - z)
        # So we can directly compute the optimal step if we know z
        # But z is unknown, so we estimate it iteratively
        
        # Use a trust-region approach with the quadratic model
        # The quadratic model near z_est is: f(x) ≈ f(z_est) + (x - z_est)^T H (x - z_est) / 2
        # where H ≈ 2I (for sphere)
        
        # Since we know the Hessian is approximately 2I for sphere,
        # we can take Newton steps: x_new = x - H^{-1} grad
        # H^{-1} = I/2, so x_new = x - grad/2
        
        # But we need to estimate grad at the current point
        # Let's do a few iterations of gradient estimation and Newton steps
        
        x_current = z_est.copy()
        f_current = best_fx
        
        # Number of refinement iterations
        max_refine_iters = min(50, (budget - call_count) // (2 * dim + 1))
        
        for it in range(max_refine_iters):
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Estimate gradient at x_current using central differences
            grad_local = np.zeros(dim)
            
            # Adaptive step size for gradient estimation
            step_size = 0.1 * (1.0 / (1.0 + 0.1 * it))  # Decreasing step
            
            for i in range(dim):
                if call_count >= budget or time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_plus = x_current + step_size * e_i
                x_minus = x_current - step_size * e_i
                
                # Clip to bounds
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = call_func(x_plus)
                f_minus = call_func(x_minus)
                
                grad_local[i] = (f_plus - f_minus) / (2.0 * step_size)
            
            # Check if we've converged
            if np.linalg.norm(grad_local) < 1e-8:
                break
            
            # Newton step: x_new = x - H^{-1} grad
            # For sphere, H ≈ 2I, so H^{-1} = I/2
            # x_new = x - grad/2
            x_new = x_current - grad_local / 2.0
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = call_func(x_new)
            
            # Line search: if the step didn't improve, try a smaller step
            if f_new >= f_current:
                # Backtracking line search
                alpha = 0.5
                for _ in range(5):
                    if call_count >= budget or time.time() - start_time > timeout:
                        break
                    x_trial = x_current - alpha * grad_local / 2.0
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = call_func(x_trial)
                    if f_trial < f_current:
                        x_new = x_trial
                        f_new = f_trial
                        break
                    alpha *= 0.5
            
            x_current = x_new
            f_current = f_new
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
