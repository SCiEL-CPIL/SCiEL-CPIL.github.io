"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def nelder_mead(x0, max_iter):
            nonlocal calls_made
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            calls_made += 1
            
            # Create other points in simplex
            for i in range(d):
                x_new = x0.copy()
                x_new[i] += 0.1 # Small step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f_simplex.append(func(x_new))
                calls_made += 1
            
            # Nelder-Mead parameters
            alpha = 0.5
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for it in range(max_iter):
                if calls_made >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                
                # Check convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-8:
                    break
                
                # Centroid of best d points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + alpha * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - alpha * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= budget:
                                return simplex[0], f_simplex[0]
            
            return simplex[0], f_simplex[0]

        # Global phase: Generate K initial points
        n_starts = 10
        budget_per_start = (budget - n_starts) // n_starts
        
        # Generate initial points using stratified sampling
        # Divide [-5, 5] into 2^d cells? No, just random.
        initial_points = []
        for i in range(n_starts):
            x = rng.uniform(-5, 5, d)
            initial_points.append(x)
        
        # Evaluate initial points
        for x in initial_points:
            if calls_made >= budget:
                break
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Local phase: Run Nelder-Mead from each initial point
        for x in initial_points:
            if calls_made >= budget - 5:
                break
            # Run Nelder-Mead with remaining budget
            local_x, local_f = nelder_mead(x, 100)
            if local_f < best_f:
                best_f = local_f
                best_x = local_x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
