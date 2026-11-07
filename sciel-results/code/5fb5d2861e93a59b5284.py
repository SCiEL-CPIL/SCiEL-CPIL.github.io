import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        
        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            return f

        # Initialize simplex
        # Center at 0, step size 1.0
        step = 1.0
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim):
            simplex[i, i] = step
        # simplex[0] is zeros, simplex[i] has step at i-th position
        
        # Evaluate initial simplex
        simplex_f = np.zeros(dim + 1)
        for i in range(dim + 1):
            f_val = safe_func(simplex[i])
            simplex_f[i] = f_val
            if f_val < best_f:
                best_f = f_val
                best_x = simplex[i].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iters = budget - calls_used
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Sort simplex by function values
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            # Check convergence
            if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                break
                
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_func(x_r)
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_func(x_c)
                
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_shrink = safe_func(simplex[i])
                        simplex_f[i] = f_shrink
                        if f_shrink < best_f:
                            best_f = f_shrink
                            best_x = simplex[i].copy()
                            
            # Update best after each step
            min_f_in_simplex = np.min(simplex_f)
            if min_f_in_simplex < best_f:
                best_f = min_f_in_simplex
                best_x = simplex[np.argmin(simplex_f)].copy()

        return best_x, best_f
