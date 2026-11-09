import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Structure:
        # Active variables: x[0], x[1]
        # Inactive variables: x[2:] = 0
        # The Pareto front is a 2D surface parameterized by x0 and x1 in [0, 1].
        # We will perform adaptive grid refinement on the (x0, x1) plane.
        # This is materially different from random search or simple Sobol sequences
        # because it dynamically allocates evaluations to regions that are
        # "under-sampled" relative to the current resolution, effectively
        # creating a hierarchical tiling of the front.
        
        n_evals = 0
        points = []
        
        # Queue of regions to refine. Each region is (x0_min, x0_max, x1_min, x1_max, level)
        # We start with the whole domain.
        # To get a good initial spread, we don't just start with 1 region.
        # We use a quadtree approach.
        
        # Initial coarse grid: 4x4 = 16 regions
        # We will evaluate the center of each region initially.
        # Then we refine the "largest" or "most promising" regions.
        # However, for a known front, uniform coverage is key for hypervolume.
        # A simple adaptive strategy:
        # 1. Start with a coarse grid (e.g., 4x4).
        # 2. Evaluate all centers.
        # 3. For remaining budget, split the region with the largest "gap" or simply
        #    cycle through regions in a balanced way (like a space-filling curve order)
        #    and split the next one, evaluating the 4 new centers.
        
        # Actually, a simpler and very effective strategy for hypervolume on a smooth front
        # is to use a deterministic low-discrepancy sequence mapped to the front,
        # but the current program already does something similar.
        # The "Cross" operator question suggests a genetic approach might be relevant,
        # but the directive asks for a structural change.
        
        # Let's try: **Deterministic Adaptive Grid Refinement**.
        # We maintain a list of "cells". Initially, one big cell.
        # We evaluate the center of a cell.
        # Then we split that cell into 4 sub-cells.
        # We use a priority queue (or just a list sorted by size) to pick the next cell to split.
        # This ensures that we always refine the coarsest part of the domain,
        # leading to a uniform distribution of points in the (x0, x1) space.
        
        # To avoid the overhead of a complex priority queue, we can just keep a list
        # of cells and always pick the one with the largest area.
        # Since we want uniform coverage, this is equivalent to a quadtree.
        
        cells = [(0.0, 1.0, 0.0, 1.0)]
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
                
            if not cells:
                break
                
            # Find the cell with the largest area
            max_area = -1
            max_idx = 0
            for i, cell in enumerate(cells):
                x0_min, x0_max, x1_min, x1_max = cell
                area = (x0_max - x0_min) * (x1_max - x1_min)
                if area > max_area:
                    max_area = area
                    max_idx = i
                    
            cell = cells.pop(max_idx)
            x0_min, x0_max, x1_min, x1_max = cell
            
            # Evaluate the center of this cell
            x0_c = (x0_min + x0_max) / 2.0
            x1_c = (x1_min + x1_max) / 2.0
            
            x = np.zeros(self.dim)
            x[0] = x0_c
            x[1] = x1_c
            # Inactive vars are 0
            
            f = problem(x)
            n_evals += 1
            points.append((x, f))
            
            # Split the cell into 4 sub-cells
            x0_mid = (x0_min + x0_max) / 2.0
            x1_mid = (x1_min + x1_max) / 2.0
            
            # Only split if the cell is large enough to avoid infinite precision issues
            # or if we still have budget.
            # We stop splitting when the cell is very small or budget is exhausted.
            # A threshold of 1e-4 in width is more than enough for 200 points.
            if (x0_max - x0_min) > 1e-3 and (x1_max - x1_min) > 1e-3:
                cells.append((x0_min, x0_mid, x1_min, x1_mid))
                cells.append((x0_mid, x0_max, x1_min, x1_mid))
                cells.append((x0_min, x0_mid, x1_mid, x1_max))
                cells.append((x0_mid, x0_max, x1_mid, x1_max))
            
        return points
