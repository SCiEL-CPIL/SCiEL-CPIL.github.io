# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Nelder-Mead Simplex Optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Phase 1: Coarse Grid Search to identify promising basins.
       Since f7 is a step ellipsoidal function, the landscape has a specific structure
       where the minimum is in a specific rotated/shifted location. A coarse grid 
       ensures we find the general vicinity of the minimum quickly.
    2. Phase 2: Nelder-Mead Simplex Local Optimization.
       Initialize a simplex around the best point from Phase 1.
       Run Nelder-Mead to converge to the minimum.
       Nelder-Mead is robust to the non-smooth "step" nature if initialized close enough,
       and exploits the local convexity for fast convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_f = np.inf
        best_x = None

        def wrapped(x):
            nonlocal call_count, best_f, best_x
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # Phase 1: Coarse Grid Search
        # We need to find the basin. A 3^5 grid = 243 points.
        # This is affordable within 500 budget.
        # We use a 3-level grid: -5, 0, 5
        grid_points = []
        levels = [-5.0, 0.0, 5.0]
        for x0 in levels:
            for x1 in levels:
                for x2 in levels:
                    for x3 in levels:
                        for x4 in levels:
                            grid_points.append(np.array([x0, x1, x2, x3, x4]))
        
        # Shuffle grid points to avoid systematic bias if the function is weird
        np.random.shuffle(grid_points)
        
        for x in grid_points:
            if call_count >= budget or time.time() > deadline:
                break
            wrapped(x)
            
        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead Local Optimization
        # Initialize simplex around best_x
        # Standard Nelder-Mead uses dim+1 points.
        # Initial step size: 0.1 is usually good for [-5,5] box if we've found the basin.
        # Let's use a slightly larger step to ensure we cover the local basin quickly.
        step_size = 0.5
        
        x0 = best_x.copy()
        simplex = [x0]
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += step_size
            # Clip to box
            x_i = np.clip(x_i, -5.0, 5.0)
            simplex.append(x_i)
            
        # Evaluate initial simplex
        for x in simplex:
            if call_count >= budget or time.time() > deadline:
                break
            wrapped(x)
            
        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        max_iter = 100 # Max iterations per simplex run
        
        # We might need to restart if NM gets stuck or converges to a local min
        # But for f7, the basin is typically convex.
        # We will run NM until convergence or budget/time limit.
        
        for _ in range(max_iter):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Sort simplex by function value
            # We need to store function values for the simplex
            # To avoid re-evaluating, we keep track of values
            # However, wrapped() doesn't return a history easily, so we maintain a local list
            # Let's re-implement the NM loop carefully to manage evaluations.
            
            # Actually, it's safer to manage the simplex state explicitly.
            # Current simplex points and their values
            # We just evaluated them above, but we didn't store the values in a list corresponding to points.
            # Let's re-evaluate the current simplex to get values? No, that wastes budget.
            # We should have stored them.
            
            # Let's restart the NM logic properly with state tracking.
            # The previous loop evaluated simplex points.
            # We need to map x -> f(x).
            # Since wrapped updates global best, we can't easily get the specific f for each simplex vertex
            # unless we re-evaluate or store it.
            # Re-evaluating is wasteful.
            # Let's assume the initial simplex evaluation happened.
            # We need to store the values.
            
            # Correction: The loop above `for x in simplex: wrapped(x)` did not store values.
            # We need to modify the approach to store values for the simplex vertices.
            
            # Let's break and re-do the NM setup with proper value tracking.
            # Since we already spent calls, we just proceed with a fresh NM instance
            # that tracks its own values, re-evaluating the center if necessary?
            # No, better to just implement a clean NM that tracks values.
            
            # Let's abandon the partially started NM and start a clean one,
            # accepting the cost of re-evaluating the initial simplex points (dim+1 calls).
            # This is acceptable if it ensures correctness.
            
            break # Exit the broken loop

        # Clean NM Implementation
        # Initialize simplex
        x0 = best_x.copy()
        simplex = [x0]
        step_size = 0.5
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += step_size
            x_i = np.clip(x_i, -5.0, 5.0)
            simplex.append(x_i)
            
        simplex_vals = []
        for x in simplex:
            if call_count >= budget or time.time() > deadline:
                return best_x, best_f
            simplex_vals.append(wrapped(x))
            
        # NM Loop
        for _ in range(200): # Max NM iterations
            if call_count >= budget or time.time() > deadline:
                break
                
            # Sort simplex
            indices = np.argsort(simplex_vals)
            simplex = [simplex[i] for i in indices]
            simplex_vals = [simplex_vals[i] for i in indices]
            
            # Check convergence
            if simplex_vals[-1] - simplex_vals[0] < 1e-6:
                break
                
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = wrapped(x_r) if (call_count < budget and time.time() <= deadline) else np.inf
            
            if f_r < simplex_vals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = wrapped(x_e) if (call_count < budget and time.time() <= deadline) else np.inf
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_vals[-1] = f_r
            elif f_r < simplex_vals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                simplex_vals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = wrapped(x_c) if (call_count < budget and time.time() <= deadline) else np.inf
                
                if f_c < simplex_vals[-1]:
                    simplex[-1] = x_c
                    simplex_vals[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_vals[i] = wrapped(simplex[i]) if (call_count < budget and time.time() <= deadline) else np.inf
                        
                    if call_count >= budget or time.time() > deadline:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
