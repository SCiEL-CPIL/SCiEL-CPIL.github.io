import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Initialize at origin
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Initial step size
        h = 0.5
        
        # Track best
        x_best = x.copy()
        fx_best = fx
        
        # Phase 1: Coarse exploration with larger steps
        # Use stochastic gradient descent with finite differences
        # Estimate gradient using forward differences with random directions
        
        while calls < budget:
            # Estimate gradient using finite differences
            # Use central differences for better accuracy
            g = np.zeros(dim)
            
            for i in range(dim):
                if calls + 2 > budget:
                    break
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                g[i] = (f_plus - f_minus) / (2 * h)
            
            # Check if we've exhausted budget
            if calls >= budget:
                break
            
            # Check gradient norm for convergence
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            
            # Estimate Hessian using finite differences on random perturbations
            # Use a simplified approach: estimate diagonal Hessian only for stability
            H_diag = np.zeros(dim)
            
            for i in range(dim):
                if calls + 2 > budget:
                    break
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                H_diag[i] = (f_plus - 2 * fx + f_minus) / (h * h)
            
            if calls >= budget:
                break
            
            # Compute Newton-like step using diagonal Hessian (quasi-Newton approximation)
            # step_i = -g_i / H_diag_i, with regularization
            step = np.zeros(dim)
            for i in range(dim):
                if abs(H_diag[i]) > 1e-8:
                    step[i] = -g[i] / H_diag[i]
                else:
                    step[i] = -g[i]
            
            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > h:
                step = step * (h / step_norm)
            
            # Line search
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            found_alpha = False
            best_trial_f = fx
            best_trial_x = x.copy()
            
            for _ in range(15):
                if calls + 1 > budget:
                    break
                x_trial = x + alpha * step
                f_trial = func(x_trial)
                calls += 1
                
                # Track best trial
                if f_trial < best_trial_f:
                    best_trial_f = f_trial
                    best_trial_x = x_trial.copy()
                
                # Armijo condition
                if f_trial < fx + c1 * alpha * np.dot(g, step):
                    found_alpha = True
                    break
                alpha *= rho
            
            if calls >= budget:
                break
            
            if found_alpha:
                x = x_trial
                fx = f_trial
            else:
                # Use best trial found
                if best_trial_f < fx:
                    x = best_trial_x
                    fx = best_trial_f
                else:
                    # No improvement, reduce step size and continue
                    h *= 0.5
                    if h < 1e-8:
                        break
                    continue
            
            # Update best
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
            
            # Adaptive step size
            if found_alpha:
                h = max(h * 0.8, 1e-6)
            else:
                h *= 0.5
            
            # Convergence check
            if np.linalg.norm(step) < 1e-10:
                break
        
        return x_best, fx_best
