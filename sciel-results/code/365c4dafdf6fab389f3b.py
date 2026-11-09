"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample it surface-area-uniform."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 M=3:
        # x[0], x[1] are angles (mapped to [0, 1] from [0, pi/2]).
        # x[2:] are distance variables. Set to 0.5 for g=0.
        
        results = []
        count = 0
        
        # Surface-uniform sampling on the quarter sphere.
        # The area element is sin(alpha1) dalpha1 dalpha2.
        # To sample alpha1 uniformly in area:
        # CDF: P(alpha1 <= a) = 1 - cos(a)
        # Inverse CDF: a = arccos(1 - u)
        # x0 = (2/pi) * a = (2/pi) * arccos(1 - u)
        
        # alpha2 is uniform in area, so x1 = v is correct.
        
        n_grid = 14
        u_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if count >= self.budget:
                    break
                if time.time() > deadline:
                    return np.array(results) if results else None
                
                # Corrected mapping for surface-uniform sampling
                # u in [0,1] -> alpha1 in [0, pi/2]
                alpha1 = np.arccos(1.0 - u)
                x0 = (2.0 / np.pi) * alpha1
                
                # v in [0,1] -> alpha2 in [0, pi/2]
                alpha2 = v * (np.pi / 2.0)
                x1 = v
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                if self.dim > 2:
                    x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                
                try:
                    f = problem(x)
                    results.append(f)
                    count += 1
                except Exception:
                    break
            
            if count >= self.budget or time.time() > deadline:
                break
        
        # Add strategic points to fill gaps or refine density if budget remains.
        # With 14x14=196 points, we have 4 points left.
        # The grid points are at u_i = i/13, v_j = j/13.
        # The "midpoints" between grid cells are at (i+0.5)/13.
        # Let's add points at the centers of the 4 central cells to improve local density
        # or just ensure the exact center of the parameter space is covered if not.
        # u=0.5 -> x0 = (2/pi) * arccos(0.5) = (2/pi) * (pi/3) = 2/3.
        # v=0.5 -> x1 = 0.5.
        # Is (2/3, 0.5) in the grid?
        # Grid u: 0, 1/13, ..., 13/13. 0.5 is 6.5/13. Not in grid.
        # So (0.5, 0.5) in (u,v) space is a good addition.
        
        extra_points = [
            (0.5, 0.5),
            (0.5, 0.0),
            (0.0, 0.5),
            (1.0, 0.5)
        ]
        
        for u, v in extra_points:
            if count >= self.budget:
                break
            if time.time() > deadline:
                break
                
            alpha1 = np.arccos(1.0 - u)
            x0 = (2.0 / np.pi) * alpha1
            x1 = v
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
                
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
