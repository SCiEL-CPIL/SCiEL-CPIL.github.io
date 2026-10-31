"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def construct():
    # Use Hungarian algorithm to match circles to grid cells
    # Formulate as: for a given grid and aspect ratio, find optimal assignment
    # that maximizes the minimum radius while enforcing non-overlap
    
    best_result = None
    best_sum = -1
    
    # Try different grid resolutions and aspect ratios
    for grid_res in [15, 20, 25]:
        for w_frac in [0.4, 0.45, 0.5, 0.55, 0.6]:
            w = 2 * w_frac
            h = 2 - w
            
            # Create grid
            cell_w = w / grid_res
            cell_h = h / grid_res
            
            # Cell centers
            cell_xs = np.linspace(cell_w/2, w - cell_w/2, grid_res)
            cell_ys = np.linspace(cell_h/2, h - cell_h/2, grid_res)
            
            # Generate all cell positions
            cell_positions = []
            for i in range(grid_res):
                for j in range(grid_res):
                    cell_positions.append([cell_xs[i], cell_ys[j]])
            
            cell_positions = np.array(cell_positions)
            n_cells = len(cell_positions)
            
            if n_cells < N:
                continue
            
            # Compute max radius for each cell (boundary constraint)
            max_radii = np.minimum(
                np.minimum(cell_positions[:, 0], w - cell_positions[:, 0]),
                np.minimum(cell_positions[:, 1], h - cell_positions[:, 1])
            )
            
            # Create cost matrix for Hungarian algorithm
            # We want to assign N circles to N cells to maximize sum of radii
            # But we also need to enforce non-overlap.
            
            # Strategy: Use iterative refinement
            # 1. Start with a greedy assignment
            # 2. Check non-overlap
            # 3. Use Hungarian to improve
            
            # First, sort cells by radius (descending)
            sorted_indices = np.argsort(-max_radii)
            
            # Greedily select N non-overlapping cells
            selected = []
            for idx in sorted_indices:
                if len(selected) >= N:
                    break
                
                pos = cell_positions[idx]
                radius = max_radii[idx]
                
                # Check if this cell overlaps with any selected cell
                overlap = False
                for sel_idx in selected:
                    sel_pos = cell_positions[sel_idx]
                    sel_radius = max_radii[sel_idx]
                    dist = np.linalg.norm(pos - sel_pos)
                    if dist < radius + sel_radius - 1e-9:
                        overlap = True
                        break
                
                if not overlap:
                    selected.append(idx)
            
            if len(selected) < N:
                continue
            
            # Get initial positions and radii
            positions = cell_positions[selected[:N]].copy()
            radii = max_radii[selected[:N]].copy()
            
            # Use Hungarian algorithm to try to improve the assignment
            # Create a cost matrix where cost[i, j] = -max_radii[j]
            # but only for cells that are compatible with circle i
            
            # For simplicity, let's use local optimization instead
            # Start with uniform radius
            current_r = np.min(radii)
            
            # Local optimization: iteratively try to increase radius
            for opt_iter in range(50):
                # Compute minimum distance between any two circles
                min_pair_dist = np.inf
                min_pair = (0, 0)
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_pair_dist:
                            min_pair_dist = d
                            min_pair = (i, j)
                
                # Check boundary distances
                for i in range(N):
                    d_boundary = min(positions[i][0], w - positions[i][0],
                                    positions[i][1], h - positions[i][1])
                    if d_boundary < min_pair_dist:
                        min_pair_dist = d_boundary
                        min_pair = (i, -1)
                
                new_r = min_pair_dist / 2 - 1e-7
                if new_r <= current_r:
                    break
                
                # Try to move circles to increase minimum distance
                if min_pair[1] != -1:
                    i, j = min_pair
                    diff = positions[j] - positions[i]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-12:
                        direction = diff / norm
                        needed = (current_r * 2) - norm
                        if needed > 0:
                            positions[i] -= direction * needed / 2
                            positions[j] += direction * needed / 2
                else:
                    i = min_pair[0]
                    x, y = positions[i]
                    min_dist = min(x, w - x, y, h - y)
                    if x == min_dist:
                        positions[i][0] += current_r
                    elif w - x == min_dist:
                        positions[i][0] -= current_r
                    elif y == min_dist:
                        positions[i][1] += current_r
                    else:
                        positions[i][1] -= current_r
                
                # Ensure circles stay within bounds
                for i in range(N):
                    positions[i][0] = np.clip(positions[i][0], current_r, w - current_r)
                    positions[i][1] = np.clip(positions[i][1], current_r, h - current_r)
                
                current_r = new_r
            
            # Final validation
            valid = True
            for i in range(N):
                if positions[i][0] < current_r - 1e-9 or positions[i][0] > w - current_r + 1e-9:
                    valid = False
                    break
                if positions[i][1] < current_r - 1e-9 or positions[i][1] > h - current_r + 1e-9:
                    valid = False
                    break
            
            if not valid:
                continue
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 2 * current_r - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            total_sum = N * current_r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = current_r
                best_result = result
    
    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
