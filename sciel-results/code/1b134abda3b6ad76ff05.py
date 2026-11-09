"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # 1. Multi-start grid to handle rotation/shift
        # 5D space, 3 points per dim = 243 starts. 
        # This is a structural change from random single start to systematic exploration.
        grid_points = np.array(np.meshgrid(*[np.linspace(-4.5, 4.5, 3) for _ in range(self.dim)])).T.reshape(-1, self.dim)
        
        best_x = grid_points[0]
        best_f = float('inf')
        
        # Initial evaluation of grid
        for x0 in grid_points:
            if time.time() > deadline:
                break
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0
        
        # 2. Nelder-Mead refinement from best points
        # We will run NM from the top few best grid points to ensure robustness
        # but strictly track budget.
        
        # Sort grid points by function value to pick best starting points for NM
        # Note: We already have best_f, but let's collect a few good ones if budget allows
        # Actually, running NM on all 243 is too expensive. 
        # Strategy: Run NM on the single best point found in the grid.
        # If it gets stuck or takes too long, we might try a second best.
        # But Rosenbrock is unimodal (in the valley sense), so one good start is usually enough.
        
        # Let's pick the best 5 grid points to run NM on, in case the global min is in a different basin 
        # (though Rosenbrock f8 has one global min, the rotation/shift might make the landscape tricky if bounds clip it).
        # Actually, the problem states "minimizing func on [-5, 5]^5".
        
        # To be safe and fast, we will run Nelder-Mead on the best grid point.
        # We need a budget-aware Nelder-Mead.
        
        def nelder_mead(x0, f0, max_iter=100):
            # Initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Expand simplex
            for i in range(self.dim):
                x_new = x0.copy()
                # Perturb slightly
                if x_new[i] < 4.9:
                    x_new[i] += 0.5
                else:
                    x_new[i] -= 0.5
                f_new = func(x_new)
                simplex.append(x_new)
                simplex_f.append(f_new)
            
            # Standard NM loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_r)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                        
                        if time.time() > deadline:
                            break
            
            # Return best from this simplex
            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx]

        # Run NM on the best grid point
        x_nm, f_nm = nelder_mead(best_x, best_f)
        
        if f_nm < best_f:
            best_x = x_nm
            best_f = f_nm
            
        return best_x, best_f
# EVOLVE-BLOCK-END
