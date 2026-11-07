import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        h = 1e-4
        
        while calls < self.budget:
            # 1. Estimate Gradient and Diagonal Hessian
            g = np.zeros(dim)
            H = np.zeros((dim, dim))
            
            f_center = fx # We have fx from previous iteration or init
            
            for i in range(dim):
                if calls + 2 > self.budget:
                    break
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                g[i] = (f_plus - f_minus) / (2 * h)
                H[i, i] = (f_plus - 2 * f_center + f_minus) / (h * h)
            
            # Check convergence
            if np.linalg.norm(g) < 1e-10:
                break
            
            # 2. Estimate Off-diagonal Hessian
            for i in range(dim):
                for j in range(i + 1, dim):
                    if calls + 4 > self.budget:
                        break
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    f_pp = func(x_pp)
                    f_pm = func(x_pm)
                    f_mp = func(x_mp)
                    f_mm = func(x_mm)
                    calls += 4
                    
                    H_val = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    H[i, j] = H_val
                    H[j, i] = H_val
            
            # 3. Newton Step
            # Regularize Hessian if not PD
            H_reg = H.copy()
            eigvals = np.linalg.eigvalsh(H_reg)
            if np.min(eigvals) < 1e-6:
                H_reg += (1e-6 - np.min(eigvals) + 1e-12) * np.eye(dim)
                
            try:
                step = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                step = -g # Fallback to gradient descent
            
            # 4. Line Search
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            found_alpha = False
            
            for _ in range(10):
                if calls + 1 > self.budget:
                    break
                x_trial = x + alpha * step
                f_trial = func(x_trial)
                calls += 1
                
                # Armijo condition
                if f_trial < fx + c1 * alpha * np.dot(g, step):
                    found_alpha = True
                    break
                alpha *= rho
                
            if not found_alpha:
                # If line search fails, try a very small step or break
                if alpha < 1e-8:
                    break
                # Use the best trial point found so far? 
                # The last f_trial is the best among rejected ones? 
                # Not necessarily. 
                # Let's just break to be safe, or take the last step.
                # Taking the last step might increase f.
                # Let's break.
                break
                
            x_new = x + alpha * step
            fx_new = f_trial
            
            # Update
            x = x_new
            fx = fx_new
            
            # Convergence check on step size
            if np.linalg.norm(x_new - x) < 1e-8: # This is 0, check previous step
                pass # Need to store prev_x

        return x, fx
