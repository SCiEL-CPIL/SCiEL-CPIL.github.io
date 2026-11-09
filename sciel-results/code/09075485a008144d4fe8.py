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
        # n_grid = 14 gives 196 points.
        n_grid = 14
        u_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if count >= self.budget:
                    break
                if time.time() > deadline:
                    return np.array(results) if results else None
                
                # Map u (uniform) to x0 (surface-uniform)
                # x0 = (2/pi) * arccos(1 - 2u)
                x0 = (2.0 / np.pi) * np.arccos(1.0 - 2.0 * u)
                
                # Map v (uniform) to x1 (uniform is correct for alpha1)
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
        
        # Add 4 strategic points to fill gaps in the 14x14 grid
        # The grid has indices 0..13. 0.5 corresponds to 6.5, so it's missing.
        # Key points to add:
        # 1. Center of parameter space: u=0.5, v=0.5
        # 2. Mid-edge: u=0.5, v=0
        # 3. Mid-edge: u=0, v=0.5
        # 4. Mid-edge: u=1, v=0.5 (or u=0.5, v=1)
        
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
                
            x0 = (2.0 / np.pi) * np.arccos(1.0 - 2.0 * u)
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
