"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # 1. Define a fixed grid topology for 21 circles
    # A 5x4 grid has 20 spots. We add 1 more in the center of a 1x2 block or just use 5x5 and remove one?
    # Better: 5 rows x 4 cols = 20. Add one more? 
    # Let's use a 5x5 grid and remove the center? No, 25-4=21?
    # Let's stick to a simple 5x4 grid (20 circles) and add one? 
    # Actually, a 5x4 grid is 20 circles. We need 21.
    # Let's use a 5x5 grid (25 circles) and remove 4 corners? That leaves 21.
    # Or just use 5x5 and remove 4 specific ones to maintain symmetry?
    # Removing 4 corners from 5x5 leaves a cross-like shape or a filled center.
    # Let's try 5x5 grid, remove indices 0, 4, 20, 24 (corners).
    
    grid_size = 5
    indices = list(range(grid_size * grid_size))
    # Remove corners
    indices.remove(0)
    indices.remove(grid_size - 1)
    indices.remove(grid_size * (grid_size - 1))
    indices.remove(grid_size * grid_size - 1)
    
    assert len(indices) == N, f"Expected {N} circles, got {len(indices)}"
    
    # Initial normalized positions in [0,1]x[0,1]
    # These are centers of the grid cells
    x_norm = np.zeros(N)
    y_norm = np.zeros(N)
    for i, idx in enumerate(indices):
        row = idx // grid_size
        col = idx % grid_size
        x_norm[i] = (col + 0.5) / grid_size
        y_norm[i] = (row + 0.5) / grid_size

    # 2. Relaxation to find minimum width and height for unit radii (r=1)
    # We will iterate:
    # a. Start with r=1 for all circles.
    # b. Iteratively move circles to resolve overlaps and boundary violations.
    # c. Track the bounding box width W and height H required.
    
    # Actually, a better approach for "maximize sum of radii" with fixed topology:
    # The optimal packing for a grid-like topology is often determined by the "tightest" constraints.
    # If we fix the aspect ratio, we can scale.
    # Let's use a simple iterative solver to find the minimum bounding box for a given aspect ratio.
    
    def relax_and_get_bounds(aspect_ratio, max_iter=500, tol=1e-6):
        """
        Given an aspect ratio, find the minimum W such that circles of radius r=1
        fit in a W x H rectangle (H = W / aspect_ratio).
        Returns (W_min, x_coords, y_coords) if successful, else None.
        """
        # Initial guess: grid spacing
        # In a 5x5 grid with corners removed, the max extent is 4 units of spacing.
        # If r=1, min spacing is 2. So W ~ 4*2 + 2 = 10?
        # Let's start with a generous bound and shrink, or just relax and measure.
        
        # Start with positions centered in a box of width 10, height 10/aspect
        W_curr = 10.0
        H_curr = W_curr / aspect_ratio
        
        # Normalize positions to current box
        x = x_norm * W_curr
        y = y_norm * H_curr
        r = 1.0
        
        # Iterative relaxation
        # We want to find the minimum W such that no overlaps and all inside [0,W]x[0,H].
        # This is equivalent to: for a fixed W, is there a valid configuration?
        # Let's try to minimize the "violation" for a fixed W, and binary search W.
        
        def check_feasibility(W_target):
            H_target = W_target / aspect_ratio
            # Start from previous best or normalized grid
            x_pos = x_norm * W_target
            y_pos = y_norm * H_target
            
            # Simple iterative projection
            for _ in range(max_iter):
                moved = False
                # 1. Boundary projection
                for i in range(N):
                    if x_pos[i] < r:
                        x_pos[i] = r
                        moved = True
                    elif x_pos[i] > W_target - r:
                        x_pos[i] = W_target - r
                        moved = True
                    if y_pos[i] < r:
                        y_pos[i] = r
                        moved = True
                    elif y_pos[i] > H_target - r:
                        y_pos[i] = H_target - r
                        moved = True
                
                # 2. Overlap resolution (simple pairwise push)
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x_pos[j] - x_pos[i]
                        dy = y_pos[j] - y_pos[i]
                        dist_sq = dx*dx + dy*dy
                        min_dist = 2.0
                        if dist_sq < min_dist*min_dist - tol:
                            dist = np.sqrt(dist_sq)
                            if dist < 1e-9:
                                dx = 0.01
                                dy = 0.0
                                dist = 0.01
                            # Push apart
                            push = (min_dist - dist) / 2.0
                            x_pos[i] -= push * (dx / dist)
                            x_pos[j] += push * (dx / dist)
                            y_pos[i] -= push * (dy / dist)
                            y_pos[j] += push * (dy / dist)
                            moved = True
                            
                if not moved:
                    break
            
            # Final check
            valid = True
            for i in range(N):
                if x_pos[i] < r - tol or x_pos[i] > W_target - r + tol:
                    valid = False
                    break
                if y_pos[i] < r - tol or y_pos[i] > H_target - r + tol:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x_pos[j] - x_pos[i]
                        dy = y_pos[j] - y_pos[i]
                        if dx*dx + dy*dy < 4.0 - tol:
                            valid = False
                            break
                    if not valid:
                        break
            return valid, x_pos, y_pos

        # Binary search for minimum W
        W_low = 5.0
        W_high = 20.0
        
        # Ensure W_high is feasible
        feasible, x_h, y_h = check_feasibility(W_high)
        if not feasible:
            # Try larger
            W_high = 50.0
            feasible, x_h, y_h = check_feasibility(W_high)
            if not feasible:
                return None
        
        # Ensure W_low is infeasible (or just start low)
        for _ in range(30):
            W_mid = (W_low + W_high) / 2.0
            feasible, x_m, y_m = check_feasibility(W_mid)
            if feasible:
                W_high = W_mid
                x_best, y_best = x_m, y_m
            else:
                W_low = W_mid
                
        return W_high, x_best, y_best

    best_value = -np.inf
    best_result = None
    
    # Try a range of aspect ratios
    # Perimeter P = 2(W+H) = 4 => W+H=2.
    # H = W/aspect => W(1 + 1/aspect) = 2 => W = 2 / (1 + 1/aspect) = 2*aspect / (aspect+1)
    # Sum of radii = N * r.
    # If we scale the "unit radius" solution (r=1) to fit in W, H, the actual radius r_actual = W / W_unit.
    # Wait, if W_unit is the width required for r=1, then for a target width W_target, the max radius is r = W_target / W_unit.
    # So we want to maximize r = W_target / W_unit(aspect).
    # W_target = 2*aspect / (aspect+1).
    
    aspects = np.linspace(0.5, 2.0, 20)
    
    for aspect in aspects:
        if time.time() > deadline:
            break
            
        res = relax_and_get_bounds(aspect)
        if res is None:
            continue
            
        W_unit, x_unit, y_unit = res
        
        # Calculate actual dimensions and radius
        W_actual = 2.0 * aspect / (aspect + 1.0)
        H_actual = 2.0 - W_actual
        
        # Scale factor
        scale = W_actual / W_unit
        r_actual = scale * 1.0
        
        # Check if it fits in height too
        # H_unit = W_unit / aspect
        # H_actual = W_actual / aspect
        # scale_h = H_actual / H_unit = (W_actual / aspect) / (W_unit / aspect) = W_actual / W_unit = scale.
        # So scaling by W is consistent.
        
        # Final coordinates
        x_final = x_unit * scale
        y_final = y_unit * scale
        
        # Validation
        valid = True
        for i in range(N):
            if x_final[i] < r_actual - 1e-9 or x_final[i] > W_actual - r_actual + 1e-9:
                valid = False
                break
            if y_final[i] < r_actual - 1e-9 or y_final[i] > H_actual - r_actual + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_final[j] - x_final[i]
                    dy = y_final[j] - y_final[i]
                    if dx*dx + dy*dy < (2*r_actual)**2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
        if valid:
            value = N * r_actual
            if value > best_value:
                best_value = value
                best_result = np.column_stack([x_final, y_final, np.full(N, r_actual)])

    if best_result is None:
        # Fallback: simple grid in 1x1 square (P=4)
        # 5x5 grid, remove corners.
        # W=1, H=1.
        # Spacing = 1/5 = 0.2.
        # Max r = 0.1.
        # Sum = 21 * 0.1 = 2.1.
        # But corners removed might allow larger r?
        # In 5x5 grid, min distance is 0.2. r=0.1.
        # If we remove corners, the center circle (2,2) has neighbors at dist 0.2.
        # So r=0.1 is the limit for grid topology.
        
        x_final = np.zeros(N)
        y_final = np.zeros(N)
        r_final = 0.1
        count = 0
        for row in range(5):
            for col in range(5):
                if (row==0 and col==0) or (row==0 and col==4) or (row==4 and col==0) or (row==4 and col==4):
                    continue
                x_final[count] = (col + 0.5) * 0.2
                y_final[count] = (row + 0.5) * 0.2
                count += 1
        best_result = np.column_stack([x_final, y_final, np.full(N, r_final)])

    return best_result
# EVOLVE-BLOCK-END
