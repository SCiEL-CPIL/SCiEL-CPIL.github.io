"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Valley-tracking coordinate descent with adaptive step sizing for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        def time_left():
            return deadline - time.time()
        
        # 1. Global exploration: coarse grid to find the valley
        # The minimizer is in [-4, 4]^5. A 3^5 grid gives 243 points.
        # This is cheap and ensures we start near the valley floor.
        grid_points = np.array(np.meshgrid(*[np.linspace(-4.0, 4.0, 3) for _ in range(self.dim)])).T.reshape(-1, self.dim)
        
        best_x = grid_points[0].copy()
        best_f = float('inf')
        
        for x0 in grid_points:
            if time_left() < 10:
                break
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # 2. Local refinement: Adaptive Coordinate Descent
        # Rosenbrock has a narrow, curved valley. Standard NM or random search
        # struggles to follow the valley efficiently. Coordinate descent with
        # adaptive step sizes (backtracking line search) can follow the valley
        # walls much more efficiently by adjusting the step size in each
        # coordinate direction based on local curvature.
        
        # We will run coordinate descent from the best grid point.
        # We'll also try a few other good points if budget allows, but
        # typically the best grid point is close enough.
        
        # Adaptive Coordinate Descent
        def coordinate_descent(x0, f0, max_iter=5000):
            x = x0.copy()
            f_curr = f0
            n = len(x)
            
            # Initial step sizes
            steps = np.ones(n) * 0.5
            
            for it in range(max_iter):
                if time_left() < 5:
                    break
                
                improved = False
                for i in range(n):
                    if time_left() < 5:
                        break
                    
                    # Try positive step
                    x_new = x.copy()
                    x_new[i] += steps[i]
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = func(x_new)
                    
                    if f_new < f_curr:
                        # Accept and try to expand step in this direction
                        x = x_new
                        f_curr = f_new
                        steps[i] = min(steps[i] * 1.5, 10.0)
                        improved = True
                        continue
                    else:
                        # Try negative step
                        x_new = x.copy()
                        x_new[i] -= steps[i]
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        
                        if f_new < f_curr:
                            # Accept and try to expand step in this direction
                            x = x_new
                            f_curr = f_new
                            steps[i] = min(steps[i] * 1.5, 10.0)
                            improved = True
                        else:
                            # Shrink step in this direction
                            steps[i] = max(steps[i] * 0.5, 1e-6)
                
                if not improved:
                    # If no improvement in a full cycle, break
                    break
            
            return x, f_curr

        x_cd, f_cd = coordinate_descent(best_x, best_f)
        
        if f_cd < best_f:
            best_x = x_cd
            best_f = f_cd
            
        # 3. If we still have budget, try a second run from a slightly perturbed
        #    best point to escape local minima or refine further.
        if time_left() > 30:
            # Perturb the best point slightly
            rng = np.random.default_rng(42)
            x2 = best_x + rng.normal(0, 0.1, size=self.dim)
            x2 = np.clip(x2, -5.0, 5.0)
            f2 = func(x2)
            
            if f2 < best_f:
                best_x = x2
                best_f = f2
            
            # Run CD again from this new best
            x_cd2, f_cd2 = coordinate_descent(best_x, best_f)
            if f_cd2 < best_f:
                best_x = x_cd2
                best_f = f_cd2

        return best_x, best_f
# EVOLVE-BLOCK-END
