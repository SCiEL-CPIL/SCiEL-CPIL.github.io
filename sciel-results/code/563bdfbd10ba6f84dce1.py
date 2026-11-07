"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic quadratic model fitting with local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial phase: Fit a quadratic model to find the basin
        # We use a central difference scheme on a grid to estimate the quadratic coefficients.
        # For a sphere f(x) = a||x-c||^2 + f_opt, the Hessian is constant 2a*I.
        # Gradient is 2a(x-c).
        
        # Start with a broad exploration to find a point inside the basin
        # Since the domain is [-5, 5]^5 and the minimum is in [-4, 4]^5, 
        # the origin (0,0,0,0,0) is likely close or inside the basin.
        
        x_current = np.zeros(self.dim)
        f_current = func(x_current)
        calls = 1
        
        # We will perform a sequence of local quadratic fits.
        # Each fit requires 2*dim + 1 evaluations (center + 2*dim neighbors).
        # We update the center to the predicted minimizer and repeat with a smaller step size.
        
        # Initial step size for the finite difference grid
        h = 1.0
        
        # We have budget 500.
        # Each iteration costs 2*dim + 1 = 11 calls.
        # We can afford roughly 40-50 iterations if we are careful, but we want to converge fast.
        # Let's do a few "coarse" fits with larger h, then "fine" fits with smaller h.
        
        # Strategy:
        # 1. Evaluate at center and 2*dim neighbors.
        # 2. Estimate gradient and Hessian (assuming diagonal Hessian for Sphere).
        #    For Sphere, Hessian is 2*a*I. We can estimate 'a' from the second difference.
        #    f(x+h) - 2f(x) + f(x-h) = 2*h^2 * a (for 1D slice).
        #    Actually, for f(x) = a||x-c||^2, the second derivative is 2a.
        #    So 2*h^2 * (2a) = 4*a*h^2.
        #    Let's just estimate the gradient and assume a constant curvature 'a' derived from the fit.
        
        # A simpler, robust approach for Sphere:
        # Estimate the center c using the gradient at the current point.
        # grad_f(x) = 2*a*(x-c).
        # We don't know 'a'. But we can estimate the direction to the center.
        # The center lies in the direction -grad_f(x).
        # To find the exact center, we need 'a'.
        # We can estimate 'a' by evaluating at a point x + delta * grad_dir.
        
        # Let's use the 11-point fit to get a robust local model.
        # Model: f(x) ~= f0 + g^T (x-x0) + 0.5 * (x-x0)^T H (x-x0)
        # For Sphere, H is diagonal (2a).
        
        while calls < self.budget - 20:
            if time.time() > deadline:
                break
            
            # Check if we are close to the minimum
            # If the improvement is tiny, we might be there.
            
            # Build the local model
            # Points: x0, x0 + h*e_i, x0 - h*e_i
            points = [x_current]
            f_values = [f_current]
            
            for i in range(self.dim):
                e_i = np.zeros(self.dim)
                e_i[i] = h
                
                x_plus = np.clip(x_current + e_i, -5, 5)
                x_minus = np.clip(x_current - e_i, -5, 5)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                
                points.append(x_plus)
                f_values.append(f_plus)
                points.append(x_minus)
                f_values.append(f_minus)
                
                calls += 2
                
                if calls >= self.budget:
                    break
            
            if calls >= self.budget:
                break
            
            # Fit the model
            # We assume the Hessian is diagonal: H = diag(h1, h2, ..., hd)
            # f(x) = f0 + sum(gi * xi) + sum(0.5 * hi * xi^2)
            
            # Gradient estimate: gi = (f(x+h) - f(x-h)) / (2h)
            # Curvature estimate: hi = (f(x+h) - 2f0 + f(x-h)) / h^2
            
            g_est = np.zeros(self.dim)
            h_est = np.zeros(self.dim)
            
            for i in range(self.dim):
                f_plus = f_values[1 + 2*i]
                f_minus = f_values[2 + 2*i]
                
                g_est[i] = (f_plus - f_minus) / (2 * h)
                # Second derivative is h_est[i]
                # f(x+h) = f0 + g_i h + 0.5 h_i h^2
                # f(x-h) = f0 - g_i h + 0.5 h_i h^2
                # f(x+h) - 2f0 + f(x-h) = h_i h^2
                h_est[i] = (f_plus - 2 * f_current + f_minus) / (h * h)
            
            # The local minimizer of the quadratic model is:
            # x_new = x0 - H^{-1} g
            # Since H is diagonal, x_new_i = x0_i - g_i / h_i
            
            # We must handle cases where h_i is close to 0 (flat function) or negative (max/min error)
            # For Sphere, h_i should be positive (2a).
            
            x_new = x_current.copy()
            valid = np.abs(h_est) > 1e-6
            x_new[valid] = x_current[valid] - g_est[valid] / h_est[valid]
            
            # Clip to domain
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate the new point
            f_new = func(x_new)
            calls += 1
            
            # Update best
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
                # Reduce step size for next iteration to refine
                h *= 0.5
            else:
                # If the model step didn't improve, the model might be bad or we are at the min
                # Reduce step size significantly to check local flatness
                h *= 0.2
                if h < 1e-5:
                    break

        # Final polish phase with small random steps or deterministic grid
        # Use remaining budget for a fine grid search around the best point
        remaining = self.budget - calls
        if remaining > 0:
            # Simple local search with very small steps
            h_polish = 1e-3
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Random walk with small sigma
                step = rng.normal(0, h_polish, self.dim)
                x_try = np.clip(x_current + step, -5, 5)
                f_try = func(x_try)
                calls += 1
                if f_try < f_current:
                    x_current = x_try
                    f_current = f_try
        
        return x_current, f_current
# EVOLVE-BLOCK-END
