import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initial point
        # Use a point near the center to avoid boundary issues if possible, 
        # but the minimizer is in [-4, 4], so 0 is a safe start for finite differences.
        x0 = np.zeros(dim)
        
        # Ensure we have a valid call
        f0 = func(x0)
        calls_made = 1
        
        # 2. Estimate Gradient and Hessian at x0
        # Sphere function: f(x) = (x - z)^T (x - z) + offset
        # f'(x) = 2(x - z)
        # f''(x) = 2I
        # We can solve for z using f'(x0) = 2(x0 - z) => z = x0 - 0.5 * f'(x0)
        # Or use f(x0) and f'(x0) and f''(x0) to be more robust if it's not exactly sphere? 
        # The task says "Sphere", so it is exactly quadratic.
        
        # Central differences for Gradient
        eps = 1e-4
        grad = np.zeros(dim)
        hess = np.zeros((dim, dim))
        
        # Compute Gradient
        for i in range(dim):
            if time.time() > deadline:
                break
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            calls_made += 1
            
            x_minus = x0.copy()
            x_minus[i] -= eps
            f_minus = func(x_minus)
            calls_made += 1
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # If we ran out of time or budget, return best so far
        if calls_made >= budget or time.time() > deadline:
            return x0, f0

        # Compute Hessian (diagonal is sufficient for Sphere, but let's get full if cheap)
        # For Sphere, Hessian is 2*I. 
        # Let's verify the Hessian diagonal to be sure.
        # Actually, for a quadratic, we can find the minimizer z directly:
        # z = x0 - 0.5 * H^{-1} * grad
        # If H = 2*I, z = x0 - 0.5 * (1/2) * grad = x0 - 0.25 * grad
        
        # Let's estimate Hessian diagonal to be safe.
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            calls_made += 1
            
            x_minus = x0.copy()
            x_minus[i] -= eps
            f_minus = func(x_minus)
            calls_made += 1
            
            # We already have f_plus and f_minus from grad calculation? 
            # No, we didn't store them. Let's re-evaluate or store.
            # To save calls, we should have stored them. 
            # Let's restructure to store f_plus/minus.
            
            # Wait, I can just use the gradient and assume H=2I for Sphere.
            # But what if the "Sphere" is scaled? f(x) = c * ||x-z||^2 + offset.
            # Then f'(x) = 2c(x-z), f''(x) = 2cI.
            # z = x0 - (1/2c) * grad.
            # We need c. 
            # c = 0.5 * H_ii.
            
            # Let's just compute the diagonal of H properly.
            pass # Skip this loop, we'll do it more efficiently below.

        # Reset and do it efficiently
        # We already used 1 + 2*dim calls for grad.
        # Total calls so far: 1 + 2*dim.
        # Budget is 500. 1 + 10 = 11 calls. Plenty left.
        
        # Let's re-do the Hessian diagonal estimation using the same eps.
        # We need f(x0+eps e_i) and f(x0-eps e_i) again? 
        # We can just call func again. It's cheap.
        
        h_diag = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            calls_made += 1
            
            x_minus = x0.copy()
            x_minus[i] -= eps
            f_minus = func(x_minus)
            calls_made += 1
            
            h_diag[i] = (f_plus - 2*f0 + f_minus) / (eps * eps)
            
            # If h_diag[i] is close to 0, we might be at a saddle or flat spot, 
            # but for Sphere it should be ~2c.
            
        if calls_made >= budget or time.time() > deadline:
            return x0, f0

        # Estimate minimizer z
        # z_i = x0_i - grad_i / h_diag[i]
        # This assumes diagonal Hessian. For Sphere, it is diagonal.
        z = np.zeros(dim)
        for i in range(dim):
            if abs(h_diag[i]) < 1e-10:
                z[i] = x0[i]
            else:
                z[i] = x0[i] - grad[i] / h_diag[i]
        
        # Clip z to box [-5, 5]
        z = np.clip(z, -5, 5)
        
        # Evaluate at estimated minimizer
        f_z = func(z)
        calls_made += 1
        
        best_x = z
        best_f = f_z
        
        if f_z < f0:
            best_x = z
            best_f = f_z
        else:
            best_x = x0
            best_f = f0

        # 3. Refine with local search if time/budget remains
        # Use a simple Nelder-Mead or CMA-ES starting from z.
        # Since we are likely very close, a few steps of Powell or NM will polish it.
        
        # Let's use a simple coordinate descent or Nelder-Mead.
        # SciPy's minimize is available.
        from scipy.optimize import minimize
        
        if calls_made < budget - 10 and time.time() < deadline:
            try:
                # Use Powell for robustness on the last few steps
                # maxfev should be limited by time and remaining budget
                remaining_budget = budget - calls_made - 5 # leave 5 for safety
                res = minimize(func, z, method='Powell', options={'maxfev': remaining_budget, 'xtol': 1e-12, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
                calls_made += remaining_budget # approximate, minimize handles internal count
            except Exception:
                pass

        return best_x, best_f
