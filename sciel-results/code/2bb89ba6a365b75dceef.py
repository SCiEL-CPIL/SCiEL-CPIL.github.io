import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def __call__(self, func):
        self.func = func
        self.calls = 0
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = self._safe_call(x)
        
        # Directions: standard basis
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        
        # Custom line search using quadratic interpolation
        def line_search(x, d, fx):
            # Determine valid alpha range
            alpha_min = -np.inf
            alpha_max = np.inf
            
            for i in range(self.dim):
                if abs(d[i]) < 1e-12:
                    continue
                if d[i] > 0:
                    a_min = (-5 - x[i]) / d[i]
                    a_max = (5 - x[i]) / d[i]
                else:
                    a_min = (5 - x[i]) / d[i]
                    a_max = (-5 - x[i]) / d[i]
                
                alpha_min = max(alpha_min, a_min)
                alpha_max = min(alpha_max, a_max)
            
            if alpha_min >= alpha_max:
                return x, fx, 0.0

            # Check budget before starting
            if self.calls >= self.budget:
                return x, fx, 0.0

            # Initial 3 points
            a1 = alpha_min
            a3 = alpha_max
            a2 = (a1 + a3) / 2.0
            
            if a3 - a1 < 1e-8:
                return x, fx, 0.0

            # Evaluate at 3 points
            if self.calls >= self.budget: return x, fx, 0.0
            f1 = self._safe_call(x + a1 * d)
            if self.calls >= self.budget: return x, fx, 0.0
            f2 = self._safe_call(x + a2 * d)
            if self.calls >= self.budget: return x, fx, 0.0
            f3 = self._safe_call(x + a3 * d)
            
            best_alpha = a2
            best_f = f2
            if f1 < best_f:
                best_alpha = a1
                best_f = f1
            if f3 < best_f:
                best_alpha = a3
                best_f = f3

            # Iterate quadratic fit
            for _ in range(5):
                if self.calls >= self.budget:
                    break
                
                if abs(a1 - a2) < 1e-10 or abs(a2 - a3) < 1e-10 or abs(a1 - a3) < 1e-10:
                    break
                    
                denom = 2.0 * (f1 * (a2 - a3) + f2 * (a3 - a1) + f3 * (a1 - a2))
                if abs(denom) < 1e-12:
                    break
                    
                num = f1 * (a2**2 - a3**2) + f2 * (a3**2 - a1**2) + f3 * (a1**2 - a2**2)
                alpha_new = num / denom
                alpha_new = np.clip(alpha_new, alpha_min, alpha_max)
                
                if abs(alpha_new - a1) < 1e-6 or abs(alpha_new - a2) < 1e-6 or abs(alpha_new - a3) < 1e-6:
                    break
                    
                if self.calls >= self.budget:
                    break
                    
                f_new = self._safe_call(x + alpha_new * d)
                
                if f_new < best_f:
                    best_f = f_new
                    best_alpha = alpha_new
                    
                if f_new < f1:
                    a1, f1 = alpha_new, f_new
                elif f_new < f2:
                    a2, f2 = alpha_new, f_new
                else:
                    a3, f3 = alpha_new, f_new

            x_new = x + best_alpha * d
            improvement = fx - best_f
            return x_new, best_f, improvement

        # Main loop
        while self.calls < self.budget:
            x_old = x.copy()
            fx_old = fx
            
            improvement = 0.0
            worst_dir_idx = -1
            worst_improvement = np.inf
            
            for i, d in enumerate(directions):
                if self.calls >= self.budget:
                    break
                
                x, fx, imp = line_search(x, d, fx)
                improvement += imp
                
                if imp < worst_improvement:
                    worst_improvement = imp
                    worst_dir_idx = i
            
            if self.calls >= self.budget:
                break
            
            # Check convergence
            if abs(fx_old - fx) < 1e-12:
                break
            
            # Update directions
            # New direction is x - x_old
            new_dir = x - x_old
            norm_new_dir = np.linalg.norm(new_dir)
            if norm_new_dir < 1e-12:
                continue
                
            # Line search along new_dir
            x_new_dir, fx_new_dir, imp_new = line_search(x_old, new_dir, fx_old)
            
            if self.calls >= self.budget:
                break
            
            # If the improvement along the new direction is better than the worst in the cycle,
            # replace the worst direction with the new one
            if imp_new > worst_improvement:
                # Normalize the new direction? Powell's method usually doesn't strictly require normalization
                # but it helps with step size consistency.
                directions[worst_dir_idx] = new_dir
                x, fx = x_new_dir, fx_new_dir
            else:
                # Keep x, fx from the end of the cycle
                # x, fx are already updated from the last line search in the loop
                pass

        return x, fx

    def _safe_call(self, x):
        if self.calls >= self.budget:
            # Return a large value to discourage further search?
            # But we shouldn't call if we are out.
            # The caller should check.
            pass
        self.calls += 1
        return self.func(x)
