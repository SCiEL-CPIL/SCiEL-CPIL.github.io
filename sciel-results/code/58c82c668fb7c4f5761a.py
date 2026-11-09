# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        calls_made = 0
        
        # Step 1: Evaluate at the origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0
        
        # Step 2: Evaluate along each axis to determine the gradient
        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            # Update best if this point is better
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
            # f(xi) = f0 + v_i * 5.0  => v_i = (fi - f0) / 5.0
            v[i] = (fi - f0) / 5.0
            
        # Step 3: Construct the optimal vertex based on linear model
        x_linear = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                x_linear[i] = -5.0
            elif v[i] < 0:
                x_linear[i] = 5.0
            else:
                x_linear[i] = 0.0
        
        # Step 4: Evaluate the predicted linear optimum
        if time.time() <= deadline:
            f_linear = func(x_linear)
            calls_made += 1
            if f_linear < best_f:
                best_f = f_linear
                best_x = x_linear.copy()
        
        # Step 5: Local refinement (Polish pass)
        # Use remaining budget to search locally around best_x
        # This handles cases where the function is not strictly linear
        # or has a kink that the linear model missed.
        
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Initial step size
        h = 0.5
        min_h = 1e-4
        
        # We do a few passes of coordinate search
        # If we find an improvement, we continue with smaller steps
        # If not, we assume we are at the global min and stall.
        
        while calls_made < self.budget and time.time() < deadline:
            improved = False
            
            for i in range(d):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Save original position for this dimension
                x_orig = x_curr.copy()
                
                # Try positive step
                x_plus = x_curr.copy()
                x_plus[i] += h
                if x_plus[i] > 5.0: x_plus[i] = 5.0
                if x_plus[i] < -5.0: x_plus[i] = -5.0
                
                # Check if x_plus is same as x_curr (due to clipping)
                if not np.allclose(x_plus, x_curr):
                    f_plus = func(x_plus)
                    calls_made += 1
                    
                    if f_plus < f_curr:
                        f_curr = f_plus
                        x_curr = x_plus.copy()
                        improved = True
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                
                # Try negative step from ORIGINAL position
                if calls_made < self.budget and time.time() < deadline:
                    x_minus = x_orig.copy()
                    x_minus[i] -= h
                    if x_minus[i] > 5.0: x_minus[i] = 5.0
                    if x_minus[i] < -5.0: x_minus[i] = -5.0
                    
                    if not np.allclose(x_minus, x_curr):
                        f_minus = func(x_minus)
                        calls_made += 1
                        
                        if f_minus < f_curr:
                            f_curr = f_minus
                            x_curr = x_minus.copy()
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                
                # If we improved in this dimension, we could break and restart
                # but for simplicity, we continue to next dimension.
                # The next iteration will use the new x_curr.
                
            if not improved:
                # No improvement found in a full pass.
                # We are likely at the optimum.
                break
            else:
                # Shrink step size
                h *= 0.5
                if h < min_h:
                    break

        # Step 6: Stall by re-evaluating the best point
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1
            
        return best_x, best_f
# EVOLVE-BLOCK-END
