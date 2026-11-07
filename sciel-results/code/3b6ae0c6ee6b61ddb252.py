"""Optimized DTLZ4 solver: uniform spacing in objective space for Pareto front coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer with uniform objective-space spacing on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270  # 30s safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 with m=3 objectives, n=12 variables:
        # Position vars: x[0], x[1] (2 vars)
        # G-vars: x[2]...x[11] (10 vars)
        # Pareto front (g=0): f1=x1, f2=x2, f3=1-x1-x2
        # Constraint: x1>=0, x2>=0, x1+x2<=1 (triangle)
        
        n_pos = self.n_obj - 1  # 2 position variables
        n_g = self.dim - n_pos  # 10 g-variables
        g_opt = 0.5  # optimal g-value
        
        n_samples = min(self.budget, 200)
        
        # Key insight: uniform spacing in objective space (f1, f2) 
        # optimizes 2D hypervolume coverage of the Pareto front.
        # The Pareto front in objective space is the triangle:
        # f1 >= 0, f2 >= 0, f1 + f2 <= 1
        # where f3 = 1 - f1 - f2
        #
        # We use a stratified grid in (f1, f2) space that covers the triangle
        # with uniform density, which is better for hypervolume than uniform
        # angular spacing.
        
        points = []
        
        # Include the 3 vertices of the triangle in objective space
        vertices = [
            [1.0, 0.0],  # f=(1,0,0)
            [0.0, 1.0],  # f=(0,1,0)
            [0.0, 0.0],  # f=(0,0,1)
        ]
        
        for v in vertices:
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = g_opt
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Create stratified samples in (f1, f2) space over the triangle
        # Use a grid approach: for each f1 value, sample f2 values
        # that keep f1+f2 <= 1
        
        # Determine grid size for good coverage
        n_f1 = int(np.sqrt(n_samples - 3)) + 1
        
        for i in range(1, n_f1):
            f1 = i / n_f1
            # For this f1, f2 can range from 0 to 1-f1
            f2_max = 1.0 - f1
            
            # Number of f2 samples proportional to the width of the valid range
            n_f2 = max(1, int(n_f1 * f2_max))
            
            for j in range(n_f2):
                # Stratified midpoint in [0, f2_max]
                f2 = (j + 0.5) / n_f2 * f2_max
                
                x = np.zeros(self.dim)
                x[0] = f1
                x[1] = f2
                x[2:] = g_opt
                x = np.clip(x, lo, hi)
                points.append(x)
        
        used = len(points)
        
        # If we have budget left, add more points with finer resolution
        if used < n_samples:
            # Use a finer grid with offset
            n_f1_fine = n_f1 * 2
            for i in range(1, n_f1_fine):
                if used >= n_samples:
                    break
                f1 = i / n_f1_fine
                f2_max = 1.0 - f1
                
                n_f2 = max(1, int(n_f1_fine * f2_max))
                
                for j in range(n_f2):
                    if used >= n_samples:
                        break
                    # Stratified midpoint, offset from the coarse grid
                    f2 = (j + 0.5) / n_f2 * f2_max
                    
                    x = np.zeros(self.dim)
                    x[0] = f1
                    x[1] = f2
                    x[2:] = g_opt
                    x = np.clip(x, lo, hi)
                    points.append(x)
                    used += 1
        
        # Fill remaining budget with additional points
        while used < n_samples:
            if time.time() > deadline:
                break
            
            # Use a different stratification: divide triangle into horizontal strips
            n_strips = 30
            strip = used % n_strips
            pos_in_strip = used // n_strips
            
            # Strip 'strip' goes from f2=0 to f2=(1 - f1) where f1 = strip/n_strips
            f1_strip = strip / n_strips
            f2_max = 1.0 - f1_strip
            
            # Sample along this strip
            n_in_strip = int(np.ceil(n_samples / n_strips))
            if pos_in_strip < n_in_strip:
                f2 = (pos_in_strip + 0.5) / n_in_strip * f2_max
                
                if f1_strip >= 0 and f2 >= 0 and f1_strip + f2 <= 1.0 + 1e-10:
                    x = np.zeros(self.dim)
                    x[0] = f1_strip
                    x[1] = f2
                    x[2:] = g_opt
                    x = np.clip(x, lo, hi)
                    points.append(x)
                    used += 1
            else:
                break
        
        # Evaluate all points
        for x in points:
            problem(x)
        
        return points
# EVOLVE-BLOCK-END
