import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        
        # Number of restarts
        # 5 restarts with 100 evaluations each
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        # Nelder-Mead parameters
        # Initial step size delta
        # The domain is [-5, 5], so width is 10.
        # Schwefel period is ~12.5. 
        # A delta of 1.0 is reasonable.
        delta = 1.0
        
        for r in range(n_restarts):
            # Initialize center
            x0 = rng.uniform(-4, 4, self.dim)
            
            # Initialize simplex
            # Vertices: x0, x0 + delta*e1, ..., x0 + delta*ed
            # We need dim+1 vertices.
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for i in range(self.dim):
                simplex[i+1] = x0
                simplex[i+1][i] += delta
            
            # Evaluate initial simplex
            fvals = np.array([func(v) for v in simplex])
            
            # Track best so far
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()
            
            calls_used = self.dim + 1
            budget_left = budget_per_restart - calls_used
            
            # Nelder-Mead loop
            # Standard parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            # We need to manage the budget carefully.
            # Each iteration of NM typically uses 1 evaluation (for reflection/expansion/contraction/shrink).
            # We stop when budget_left <= 0.
            
            # Sort simplex by function value
            # We maintain the simplex and fvals sorted such that simplex[0] is best.
            # This makes it easier to identify worst, second worst, etc.
            
            # Helper to sort simplex and fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            while budget_left > 0:
                # Current worst point is simplex[-1]
                # Current second worst is simplex[-2]
                # Current best is simplex[0]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                # Clip to box
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                budget_left -= 1
                
                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if budget_left <= 0:
                        break
                    f_e = func(x_e)
                    budget_left -= 1
                    
                    if f_e < f_r:
                        # Replace worst with expanded
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        # Replace worst with reflected
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    # If f_r < fvals[-1], do outside contraction, else inside
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if budget_left <= 0:
                        break
                    f_c = func(x_c)
                    budget_left -= 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        # Move all points except the best towards the best
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if budget_left <= 0:
                                break
                            fvals[i] = func(simplex[i])
                            budget_left -= 1
                
                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()
                
                # Sort simplex again for next iteration
                sort_simplex()
                
                # Optional: Check for convergence (e.g., simplex volume or function value spread)
                # For now, just rely on budget.
        
        return best_x, best_f
