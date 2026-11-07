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

        # Phase 1: Estimate the minimizer z using a 2n-point design (skipping center)
        # f(x) = ||x - z||^2 + c
        # grad_i(x0) = 2(x0_i - z_i)
        # Using central difference at x0=0:
        # g_i = (f(h e_i) - f(-h e_i)) / (2h) = 2(0 - z_i) = -2 z_i
        # z_i = -g_i / 2
        
        delta = 2.0
        grad = np.zeros(dim)
        
        for i in range(dim):
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = delta * e_i
            x_minus = -delta * e_i
            
            f_plus = call_func(x_plus)
            f_minus = call_func(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2.0 * delta)
            
        # Estimate z
        z_est = -grad / 2.0
        z_est = np.clip(z_est, -4.5, 4.5)
        
        # Evaluate estimated minimum
        if call_count < budget and time.time() - start_time <= timeout:
            f_z = call_func(z_est)
            
        # Phase 2: Local refinement using Newton's method
        # For Sphere, Hessian is 2I. Newton step: x_new = x - 0.5 * grad
        # We estimate grad at current best and take a step.
        
        x_current = best_x.copy()
        f_current = best_fx
        
        # Refine for a few iterations to ensure machine precision
        # We have plenty of budget, but we want to finish quickly.
        # 3 iterations should be enough.
        
        for it in range(3):
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Estimate gradient at x_current
            # Use a small step size
            step_size = 1e-3
            grad_local = np.zeros(dim)
            
            for i in range(dim):
                if call_count >= budget or time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_plus = x_current + step_size * e_i
                x_minus = x_current - step_size * e_i
                
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = call_func(x_plus)
                f_minus = call_func(x_minus)
                
                grad_local[i] = (f_plus - f_minus) / (2.0 * step_size)
            
            # Check convergence
            if np.linalg.norm(grad_local) < 1e-10:
                break
                
            # Newton step
            x_new = x_current - 0.5 * grad_local
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = call_func(x_new)
            
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
            else:
                # If no improvement, we might be at the minimum or numerical noise
                break

        # Phase 3: High-precision polish using exact quadratic solve
        # Since the function is exactly quadratic, we can use the Hessian H=2I
        # to take a direct step if we have an accurate gradient estimate.
        # We do a final pass with a very tight step to eliminate any residual error.
        
        if call_count < budget and time.time() - start_time <= timeout:
            # Use a much smaller step for the final gradient estimate
            step_size_final = 1e-6
            grad_final = np.zeros(dim)
            
            for i in range(dim):
                if call_count >= budget or time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_plus = x_current + step_size_final * e_i
                x_minus = x_current - step_size_final * e_i
                
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = call_func(x_plus)
                f_minus = call_func(x_minus)
                
                grad_final[i] = (f_plus - f_minus) / (2.0 * step_size_final)
            
            # If the gradient is still non-negligible, take one more Newton step
            if np.linalg.norm(grad_final) > 1e-12:
                x_polish = x_current - 0.5 * grad_final
                x_polish = np.clip(x_polish, -5.0, 5.0)
                
                if call_count < budget and time.time() - start_time <= timeout:
                    f_polish = call_func(x_polish)
                    # best_x and best_fx are updated automatically by call_func

        return best_x, best_fx

# EVOLVE-BLOCK-END
