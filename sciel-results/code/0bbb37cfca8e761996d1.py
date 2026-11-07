import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Estimate gradient and Hessian using central differences
        # Use a point near the center as base
        x0 = np.zeros(n)
        f0 = call_func(x0)
        
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Estimate gradient using central differences
        h = 0.01
        grad = np.zeros(n)
        H = np.zeros((n, n))
        
        # First order: gradient via central differences
        for i in range(n):
            if time.time() > deadline or calls >= budget:
                break
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h
            xm[i] -= h
            fp = call_func(xp)
            fm = call_func(xm)
            if calls >= budget:
                break
            grad[i] = (fp - fm) / (2 * h)
        
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Estimate Hessian using second-order central differences
        # H[i,j] = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4*h^2)
        for i in range(n):
            if time.time() > deadline or calls >= budget:
                break
            for j in range(i, n):
                if time.time() > deadline or calls >= budget:
                    break
                xpp = x0.copy()
                xpm = x0.copy()
                xmp = x0.copy()
                xmm = x0.copy()
                xpp[i] += h
                xpp[j] += h
                xpm[i] += h
                xpm[j] -= h
                xmp[i] -= h
                xmp[j] += h
                xmm[i] -= h
                xmm[j] -= h
                
                fpp = call_func(xpp)
                if time.time() > deadline or calls >= budget:
                    break
                fpm = call_func(xpm)
                if time.time() > deadline or calls >= budget:
                    break
                fmp = call_func(xmp)
                if time.time() > deadline or calls >= budget:
                    break
                fmm = call_func(xmm)
                
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                H[j, i] = H[i, j]
        
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Phase 2: Newton's method with trust region
        # For a quadratic, this should converge in 1-2 iterations
        x = x0.copy()
        f = f0
        
        # Regularize Hessian if needed
        # Add small diagonal for numerical stability
        H_reg = H + 1e-6 * np.eye(n)
        
        for newton_iter in range(20):
            if time.time() > deadline or calls >= budget:
                break
            
            # Solve H * dx = -grad
            try:
                dx = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                # If Hessian is singular, use pseudo-inverse
                try:
                    dx = np.linalg.pinv(H_reg) @ (-grad)
                except:
                    break
            
            # Trust region: limit step size
            # Check if step is too large
            step_norm = np.linalg.norm(dx)
            max_step = 10.0  # Limit step to stay in reasonable range
            if step_norm > max_step:
                dx = dx * (max_step / step_norm)
            
            x_new = x + dx
            
            # Check bounds
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = call_func(x_new)
            if time.time() > deadline or calls >= budget:
                break
            
            # Check for convergence
            if abs(f_new - f) < 1e-12:
                x = x_new
                f = f_new
                break
            
            # Update
            x = x_new
            f = f_new
            
            # Re-estimate gradient at new point if not converged
            if newton_iter < 15 and calls < budget - 10:
                h2 = 0.005
                grad_new = np.zeros(n)
                for i in range(n):
                    if time.time() > deadline or calls >= budget:
                        break
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h2
                    xm[i] -= h2
                    fp = call_func(xp)
                    fm = call_func(xm)
                    if calls >= budget:
                        break
                    grad_new[i] = (fp - fm) / (2 * h2)
                
                if calls < budget:
                    grad = grad_new
                    if np.linalg.norm(grad) < 1e-8:
                        break
            else:
                break
        
        # Phase 3: Fine-tuning with small perturbations if budget remains
        if calls < budget - 5 and time.time() < deadline:
            # Try small random perturbations around best_x
            for _ in range(min(10, budget - calls)):
                if time.time() > deadline:
                    break
                perturbation = np.random.randn(n) * 0.01
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                f_try = call_func(x_try)
                if time.time() > deadline:
                    break
        
        return best_x, best_f
