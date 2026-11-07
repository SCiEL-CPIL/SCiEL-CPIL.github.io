"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic model-based optimizer with CMA-ES fallback for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Build quadratic model using 2n+1 evaluations
        # For Sphere f(x) = ||x - z||^2 + c, we can estimate z from 2n+1 points
        # using finite differences approach
        
        # First, evaluate at origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        
        if best_x is None:
            return None, np.inf
            
        # Evaluate at x0 + h*e_i and x0 - h*e_i for each dimension
        h = 0.5
        points = [x0.copy()]
        values = [f0]
        
        for i in range(dim):
            if calls_made >= total_budget or time.time() > deadline:
                break
            xi_pos = x0.copy()
            xi_pos[i] += h
            fi_pos = safe_call(xi_pos)
            points.append(xi_pos)
            values.append(fi_pos)
            
            if calls_made >= total_budget or time.time() > deadline:
                break
            xi_neg = x0.copy()
            xi_neg[i] -= h
            fi_neg = safe_call(xi_neg)
            points.append(xi_neg)
            values.append(fi_neg)
        
        # Estimate gradient and Hessian from finite differences
        # For quadratic f(x) = (x-z)^T A (x-z) + c
        # Gradient at x0: g_i = (f(x0+h*e_i) - f(x0-h*e_i)) / (2h)
        # Hessian diagonal: H_ii = (f(x0+h*e_i) - 2*f(x0) + f(x0-h*e_i)) / h^2
        # Hessian off-diagonal: H_ij = (f(x0+h*(e_i+e_j)) - f(x0+h*(e_i-e_j))) / (4h^2)
        
        # We have 2n+1 points, so let's estimate the minimizer directly
        # For Sphere, f(x) = ||x-z||^2 + c, so z = argmin f
        # Using the finite difference points, we can set up a linear system
        
        # Actually, for Sphere specifically, we can use a simpler approach:
        # The minimizer z satisfies: z_i = x0_i + (f(x0-h*e_i) - f(x0+h*e_i)) / (2 * (f(x0+h*e_i) + f(x0-h*e_i) - 2*f0)) * h
        # This comes from the fact that for f(x) = ||x-z||^2, 
        # f(x0+h*e_i) - f(x0) = 2*h*(x0_i - z_i) + h^2
        # f(x0-h*e_i) - f(x0) = -2*h*(x0_i - z_i) + h^2
        
        # Let a_i = f(x0+h*e_i) - f0, b_i = f(x0-h*e_i) - f0
        # Then a_i = 2*h*(x0_i - z_i) + h^2, b_i = -2*h*(x0_i - z_i) + h^2
        # So a_i - b_i = 4*h*(x0_i - z_i)
        # Therefore z_i = x0_i - (a_i - b_i) / (4*h)
        
        z_est = np.zeros(dim)
        for i in range(dim):
            if 2*i + 2 < len(values) and 2*i + 3 < len(values):
                a_i = values[2*i + 1] - f0
                b_i = values[2*i + 2] - f0
                if abs(a_i - b_i) > 1e-12:
                    z_est[i] = x0[i] - (a_i - b_i) / (4 * h)
                else:
                    z_est[i] = x0[i]
            else:
                z_est[i] = x0[i]
        
        # Clip to domain
        z_est = np.clip(z_est, -5.0, 5.0)
        
        # Evaluate the estimated minimizer
        f_z = safe_call(z_est)
        
        # Phase 2: If we have budget left, do local refinement with (1+1)-ES
        if calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Smaller initial step size since we're likely close
            
            # (1+1)-ES loop
            while calls_made < total_budget and time.time() < deadline:
                # Generate offspring
                y = np.clip(x + sigma * np.random.randn(dim), -5.0, 5.0)
                fy = safe_call(y)
                
                if calls_made >= total_budget:
                    break
                    
                # Selection and adaptation
                if fy < fx:
                    # Success: move to offspring and increase step size
                    x = y
                    fx = fy
                    sigma *= 1.1  # Moderate increase
                else:
                    # Failure: decrease step size
                    sigma *= 0.7  # Decrease
                
                # Convergence check
                if sigma < 1e-10:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
