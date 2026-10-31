"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def construct():
    """
    Pack 21 circles in a rectangle with perimeter 4 (w+h=2).
    Uses bipartite matching to assign circles to grid cells,
    then optimizes positions and radii.
    """
    
    best_sum = -1
    best_result = None
    
    # Try different aspect ratios
    w_fracs = np.linspace(0.35, 0.65, 13)
    
    for w_frac in w_fracs:
        w = 2 * w_frac
        h = 2 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Try different grid resolutions
        for grid_nx in range(4, 12):
            for grid_ny in range(4, 12):
                # Skip if too few cells
                if grid_nx * grid_ny < N:
                    continue
                
                # Create grid cell centers
                cell_xs = (np.arange(grid_nx) + 0.5) * (w / grid_nx)
                cell_ys = (np.arange(grid_ny) + 0.5) * (h / grid_ny)
                
                # Generate all cell positions
                cells = []
                for i in range(grid_nx):
                    for j in range(grid_ny):
                        cells.append([cell_xs[i], cell_ys[j]])
                cells = np.array(cells)
                
                n_cells = len(cells)
                
                # Compute cost matrix: for each circle-cell pair,
                # the "cost" is negative of the maximum feasible radius
                # (we want to maximize min radius, so minimize negative)
                
                # For a circle at cell (cx, cy), max radius is limited by:
                # 1. Distance to nearest other selected cell
                # 2. Distance to boundary
                
                # First, compute boundary-limited radius for each cell
                boundary_r = np.zeros(n_cells)
                for idx, (cx, cy) in enumerate(cells):
                    boundary_r[idx] = min(cx, w - cx, cy, h - cy)
                
                # Use Hungarian algorithm to assign N circles to N cells
                # We want to select N cells that maximize the minimum pairwise distance
                # and boundary distance
                
                # Strategy: iteratively refine
                # Start with a greedy selection, then use matching to improve
                
                # Compute pairwise distances between all cells
                # For efficiency, we'll use a simpler approach:
                # Select cells using a space-filling pattern
                
                # Alternative: use the Hungarian algorithm with a cost matrix
                # that penalizes close cells
                
                # For now, let's use a simpler heuristic:
                # Sort cells by distance from center, pick well-spread ones
                
                # Compute distance from rectangle center
                center_x, center_y = w / 2, h / 2
                dist_from_center = np.sqrt((cells[:, 0] - center_x)**2 + 
                                           (cells[:, 1] - center_y)**2)
                
                # We want to select N cells that are well-spread
                # Use a greedy approach: start from corners, add farthest from selected
                
                selected = []
                remaining = list(range(n_cells))
                
                # Start with a corner cell
                corner_idx = np.argmin(dist_from_center)
                selected.append(corner_idx)
                remaining.remove(corner_idx)
                
                while len(selected) < N and remaining:
                    # Find the cell in remaining that is farthest from all selected
                    best_idx = -1
                    best_min_dist = -1
                    
                    for idx in remaining:
                        # Compute minimum distance to any selected cell
                        min_d = np.inf
                        for sel_idx in selected:
                            d = np.linalg.norm(cells[idx] - cells[sel_idx])
                            if d < min_d:
                                min_d = d
                        
                        if min_d > best_min_dist:
                            best_min_dist = min_d
                            best_idx = idx
                    
                    if best_idx == -1:
                        break
                    
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                
                if len(selected) < N:
                    continue
                
                selected = np.array(selected)
                positions = cells[selected]
                
                # Compute the maximum feasible radius
                # Limited by pairwise distances and boundary distances
                
                min_pair_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_pair_dist:
                            min_pair_dist = d
                
                min_boundary = np.min(boundary_r[selected])
                
                r = min(min_pair_dist / 2, min_boundary) - 1e-7
                
                if r <= 0:
                    continue
                
                # Local optimization: try to improve by adjusting positions
                # Simple local search: nudge circles to increase min distance
                
                for iteration in range(100):
                    # Compute gradient of min distance
                    # Find the pair that is closest
                    min_d = np.inf
                    min_pair = (-1, -1)
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_d:
                                min_d = d
                                min_pair = (i, j)
                    
                    # Also check boundary
                    for i in range(N):
                        d_b = min(positions[i][0], w - positions[i][0],
                                  positions[i][1], h - positions[i][1])
                        if d_b < min_d:
                            min_d = d_b
                            min_pair = (i, -1)  # -1 indicates boundary
                    
                    if min_pair[1] == -1:
                        # Boundary constraint is tight
                        i = min_pair[0]
                        # Move away from boundary
                        x, y = positions[i]
                        # Determine which boundary is closest
                        if x < w - x and x < y and x < h - y:
                            positions[i][0] += 0.01
                        elif w - x < x and w - x < y and w - x < h - y:
                            positions[i][0] -= 0.01
                        elif y < x and y < w - x and y < h - y:
                            positions[i][1] += 0.01
                        elif h - y < y and h - y < x and h - y < w - x:
                            positions[i][1] -= 0.01
                    else:
                        i, j = min_pair
                        # Move apart
                        diff = positions[i] - positions[j]
                        norm = np.linalg.norm(diff)
                        if norm > 1e-10:
                            diff = diff / norm
                            positions[i] += 0.005 * diff
                            positions[j] -= 0.005 * diff
                    
                    # Keep within bounds
                    for i in range(N):
                        positions[i][0] = np.clip(positions[i][0], 0.01, w - 0.01)
                        positions[i][1] = np.clip(positions[i][1], 0.01, h - 0.01)
                
                # Recompute radius after local optimization
                min_pair_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_pair_dist:
                            min_pair_dist = d
                
                min_boundary = np.inf
                for i in range(N):
                    d_b = min(positions[i][0], w - positions[i][0],
                              positions[i][1], h - positions[i][1])
                    if d_b < min_boundary:
                        min_boundary = d_b
                
                r = min(min_pair_dist / 2, min_boundary) - 1e-7
                
                if r <= 0:
                    continue
                
                total_sum = N * r
                if total_sum > best_sum:
                    best_sum = total_sum
                    result = np.zeros((N, 3))
                    result[:, 0] = positions[:, 0]
                    result[:, 1] = positions[:, 1]
                    result[:, 2] = r
                    best_result = result
    
    if best_result is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
        return np.array(cells)
    
    return best_result
# EVOLVE-BLOCK-END
