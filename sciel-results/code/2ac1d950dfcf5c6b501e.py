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
        
        # Parameters
        h = 1e-4 # Step size for finite differences
        c1 = 1e-4 # Armijo constant
        rho = 0.5 # Step size reduction
        
        # Initial Gradient
        g = self._estimate_gradient(func, x, h, dim)
        calls += 2 * dim
        
        # Initial Inverse Hessian Approximation
        B = np.eye(dim)
        
        while calls < self.budget:
            # Search Direction
            d = -B.dot(g)
            
            # Line Search
            alpha = 1.0
            # Ensure we don't exceed budget in line search
            max_ls_steps = 10
            found_alpha = False
            for _ in range(max_ls_steps):
                if calls + 1 > self.budget:
                    break
                x_trial = x + alpha * d
                # Clip to box? The func does it, but we might want to keep x in bounds for gradient estimation
                # The func clips, so x_trial can be outside, func will clip.
                # However, if x_trial is far outside, the gradient estimate might be poor due to clipping.
                # But we start at 0 and move towards min, so we should stay inside.
                f_trial = func(x_trial)
                calls += 1
                
                # Armijo Condition
                if f_trial < fx + c1 * alpha * np.dot(g, d):
                    found_alpha = True
                    break
                alpha *= rho
                
            if not found_alpha:
                # If line search fails, try a smaller step or break
                if alpha < 1e-8:
                    break
                # Use the last alpha even if condition not met? Or break.
                # Let's break to avoid wasting budget.
                break
                
            x_new = x + alpha * d
            fx_new = f_trial
            
            # Check for convergence
            if np.linalg.norm(x_new - x) < 1e-8 or np.linalg.norm(g) < 1e-8:
                x, fx = x_new, fx_new
                break
                
            # New Gradient
            g_new = self._estimate_gradient(func, x_new, h, dim)
            calls += 2 * dim
            
            if calls > self.budget:
                x, fx = x_new, fx_new
                break
                
            # BFGS Update
            s = x_new - x
            y = g_new - g
            ys = np.dot(y, s)
            
            if ys > 1e-10: # Curvature condition
                # Update B (inverse Hessian)
                # B_new = (I - s y^T / ys) B (I - y s^T / ys) + s s^T / ys
                term1 = np.eye(dim) - np.outer(s, y) / ys
                term2 = np.eye(dim) - np.outer(y, s) / ys
                B = term1.dot(B).dot(term2) + np.outer(s, s) / ys
            # Else, skip update (keep old B)
            
            x, fx, g = x_new, fx_new, g_new

        return x, fx

    def _estimate_gradient(self, func, x, h, dim):
        g = np.zeros(dim)
        for i in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            # func clips, so we don't need to clip here
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            g[i] = (f_plus - f_minus) / (2 * h)
        return g
