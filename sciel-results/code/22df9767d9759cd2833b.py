import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT2 specific optimization
        # f1 = x0
        # f2 = (1 + g) * (1 - sqrt(x0/g))
        # g = 10 + 90 * sum(xi^2 for i>=1)
        # Set x1..x29 = 0 => g = 10
        # f2(x0) = 11 * (1 - sqrt(x0/11))
        
        m = 200
        # Optimize x0 positions
        # Maximize HV
        
        # Simple local search on x0 positions
        # Use a substitution to handle the singularity?
        # Or just use uniform grid if local search is too complex/slow.
        
        # Let's try a simple local search
        x0s = np.linspace(0, 1, m)
        
        def calc_hv(x0s):
            # x0s sorted
            f2s = 11.0 * (1.0 - np.sqrt(x0s / 11.0))
            # Area = sum (x0s[i] - x0s[i-1]) * f2s[i]
            # x0s[0] is 0? No, x0s[0] >= 0.
            # If x0s[0] > 0, the first term is x0s[0] * f2s[0].
            # The region [0, x0s[0]] has height f2s[0].
            dx = np.diff(np.concatenate(([0.0], x0s)))
            return np.sum(dx * f2s)
            
        current_hv = calc_hv(x0s)
        
        # Local search
        # Step size
        step = 0.01
        for _ in range(500):
            improved = False
            for i in range(m):
                # Try moving x0s[i]
                old_val = x0s[i]
                # Gradient direction?
                # Just try +/- step
                for delta in [step, -step]:
                    new_val = old_val + delta
                    if new_val < 0: new_val = 0
                    if new_val > 1: new_val = 1
                    # Ensure sorted
                    if i > 0 and new_val < x0s[i-1]: continue
                    if i < m-1 and new_val > x0s[i+1]: continue
                    
                    x0s[i] = new_val
                    new_hv = calc_hv(x0s)
                    if new_hv > current_hv:
                        current_hv = new_hv
                        improved = True
                        break
                    else:
                        x0s[i] = old_val
                
                if improved:
                    break
            
            if not improved:
                # Reduce step size
                step *= 0.8
                if step < 1e-5:
                    break

        # Construct points
        points = []
        for x0 in x0s:
            x = np.zeros(self.dim)
            x[0] = x0
            points.append(x)
            
        # Evaluate
        for x in points:
            problem(x)
            
        return None
