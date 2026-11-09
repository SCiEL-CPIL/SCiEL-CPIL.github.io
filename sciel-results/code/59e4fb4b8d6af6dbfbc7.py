import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used, best_f, best_x
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Sampling
        # Sample 50 points in [-4.5, 4.5]^dim
        n_init = 50
        init_points = []
        for _ in range(n_init):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = eval_func(x)
            init_points.append((x, fx))
        
        # Sort by fitness
        init_points.sort(key=lambda t: t[1])
        
        # Phase 2: Multi-Start Nelder-Mead
        # Try to refine the top 3 initial points
        for i in range(3):
            if calls_used >= budget - 10:
                break
            
            if i < len(init_points):
                x0 = init_points[i][0]
            else:
                x0 = best_x.copy()
            
            # Run Nelder-Mead
            x_best, f_best = self._nelder_mead(func, x0, dim, budget - calls_used - 5, deadline, eval_func)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, max_iter, deadline, eval_func):
        # Nelder-Mead implementation
        # Initialize simplex
        simplex = []
        f_values = []
        
        # First vertex is x0
        x0 = x0.copy()
        f0 = eval_func(x0)
        simplex.append(x0)
        f_values.append(f0)
        
        # Other vertices
        step = 0.5
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += step
            x_i = np.clip(x_i, -5, 5)
            f_i = eval_func(x_i)
            simplex.append(x_i)
            f_values.append(f_i)
        
        # Sort simplex
        def sort_simplex():
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            return simplex, f_values
        
        simplex, f_values = sort_simplex()
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        best_x = simplex[0]
        best_f = f_values[0]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Calculate centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = eval_func(x_r)
            
            if f_r < f_values[-2]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = eval_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = eval_func(x_c)
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = eval_func(simplex[i])
            
            # Sort
            simplex, f_values = sort_simplex()
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
            
            # Convergence check
            if np.std(f_values) < 1e-8:
                break
        
        return best_x, best_f
