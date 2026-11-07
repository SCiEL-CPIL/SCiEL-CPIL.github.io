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
            
        # Phase 2: Local refinement using direct quadratic solve
        # For Sphere: f(x) = ||x - z||^2 + c
        # If we know f at x0 and at x0 + h*e_i, we can solve for z_i directly:
        # f(x0 + h*e_i) - f(x0) = 2*h*(x0_i - z_i) + h^2
        # => z_i = x0_i - (f(x0+h*e_i) - f(x0) - h^2) / (2*h)
        
        x_current = best_x.copy()
        f_current = best_fx
        
        # Adaptive step sizes for refinement
        step_sizes = [1.0, 0.1, 0.01, 0.001, 0.0001]
        
        for step_size in step_sizes:
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # First, evaluate at current point if not already done
            # (best_fx is already evaluated)
            
            z_local = np.zeros(dim)
            valid = True
            
            for i in range(dim):
                if call_count >= budget or time.time() - start_time > timeout:
                    valid = False
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_offset = x_current + step_size * e_i
                x_offset = np.clip(x_offset, -5.0, 5.0)
                
                f_offset = call_func(x_offset)
                
                # Solve for z_i using the quadratic relation
                # f(x0 + h*e_i) - f(x0) = 2*h*(x0_i - z_i) + h^2
                # z_i = x0_i - (f_offset - f_current - h^2) / (2*h)
                z_i = x_current[i] - (f_offset - f_current - step_size**2) / (2.0 * step_size)
                z_local[i] = z_i
            
            if not valid:
                break
            
            # Clip z estimate to valid range
            z_local = np.clip(z_local, -4.9, 4.9)
            
            # Evaluate the estimated minimum
            if call_count < budget and time.time() - start_time <= timeout:
                f_z = call_func(z_local)
                
                if f_z < f_current:
                    x_current = z_local.copy()
                    f_current = f_z
                else:
                    # If no improvement, we're likely at or near the minimum
                    # Try a smaller step if we haven't used them all
                    continue
            
            # If we improved, continue with smaller steps
            if f_current <= best_fx:
                continue
            else:
                break
        
        # Phase 3: Final high-precision polish
        # Use a very small step for the final gradient-free solve
        if call_count < budget and time.time() - start_time <= timeout:
            step_final = 1e-4
            z_final = np.zeros(dim)
            
            for i in range(dim):
                if call_count >= budget or time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_offset = x_current + step_final * e_i
                x_offset = np.clip(x_offset, -5.0, 5.0)
                
                f_offset = call_func(x_offset)
                
                z_i = x_current[i] - (f_offset - f_current - step_final**2) / (2.0 * step_final)
                z_final[i] = z_i
            
            z_final = np.clip(z_final, -4.9, 4.9)
            
            if call_count < budget and time.time() - start_time <= timeout:
                f_final = call_func(z_final)
                # best_x and best_fx are updated automatically by call_func

        return best_x, best_fx

# EVOLVE-BLOCK-END
