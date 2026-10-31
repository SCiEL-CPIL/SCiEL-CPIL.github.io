"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def construct():
    # Formulate packing as minimum-weight matching in bipartite graph
    # Left side: N circles, Right side: grid cells
    # Weight: negative of feasible radius if circle can fit in cell
    # Use Hungarian algorithm to find optimal assignment
    
    best_result = None
    best_sum = -1
    
    # Try different grid resolutions and aspect ratios
    for nx in range(4, 12):
        for ny in range(4, 12):
            if nx * ny < N:
                continue
            
            for w_frac in np.linspace(0.3, 0.7, 9):
                w = 2 * w_frac
                h = 2 - w
                
                # Create high-resolution grid
                # Cell centers
                cell_xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
                cell_ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
                
                # Generate all cell positions
                cells = []
                for i in range(nx):
                    for j in range(ny):
                        cells.append([cell_xs[i], cell_ys[j]])
                cells = np.array(cells)
                n_cells = len(cells)
                
                # For each circle-cell pair, compute maximum feasible radius
                # A circle at cell (cx, cy) with radius r must satisfy:
                # r <= min(cx, w-cx, cy, h-cy)  [boundary constraint]
                # And for non-overlap, we need r <= dist/2 for all other circles
                # But in matching, we approximate by using cell-based spacing
                
                # Compute boundary-limited radius for each cell
                cell_r_max = np.min([
                    cells[:, 0], w - cells[:, 0],
                    cells[:, 1], h - cells[:, 1]
                ], axis=1)
                
                # Estimate minimum spacing between adjacent cells
                # For a regular grid, min spacing is min(w/nx, h/ny)
                min_spacing = min(w / nx, h / ny)
                
                # Build cost matrix: N circles x n_cells
                # Cost = -r_max[cell] (we want to maximize radius, so minimize negative)
                # But we also need to ensure circles don't overlap
                # Approximation: use cell radius as proxy
                
                # Create cost matrix
                cost_matrix = np.zeros((N, n_cells))
                for i in range(N):
                    for j in range(n_cells):
                        # Feasible radius is limited by cell position and spacing
                        r_feasible = min(cell_r_max[j], min_spacing / 2 - 1e-6)
                        if r_feasible > 0:
                            cost_matrix[i, j] = -r_feasible
                        else:
                            cost_matrix[i, j] = 1e6  # Infeasible
                
                # Use Hungarian algorithm to find optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Extract assigned positions and radii
                positions = np.zeros((N, 2))
                radii = np.zeros(N)
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < 1e5:  # Valid assignment
                        positions[i] = cells[j]
                        radii[i] = -cost_matrix[i, j]
                    else:
                        # Fallback: place at cell center with small radius
                        positions[i] = cells[j]
                        radii[i] = min(cell_r_max[j], min_spacing / 4 - 1e-6)
                
                # Local optimization: try to increase radii
                # Iterate to find better packing
                for iteration in range(30):
                    # Compute minimum distance between any two circles
                    min_pair_dist = np.inf
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_pair_dist:
                                min_pair_dist = d
                    
                    # Check boundary constraints
                    for i in range(N):
                        d_boundary = min(positions[i][0], w - positions[i][0],
                                        positions[i][1], h - positions[i][1])
                        if d_boundary < min_pair_dist:
                            min_pair_dist = d_boundary
                    
                    # Update radius
                    new_r = min_pair_dist / 2 - 1e-7
                    if new_r <= 0:
                        break
                    
                    # Check if all circles can have this radius
                    valid = True
                    for i in range(N):
                        max_r = min(positions[i][0], w - positions[i][0],
                                   positions[i][1], h - positions[i][1])
                        if new_r > max_r + 1e-9:
                            valid = False
                            break
                    
                    if not valid:
                        # Reduce radius to fit boundaries
                        for i in range(N):
                            max_r = min(positions[i][0], w - positions[i][0],
                                       positions[i][1], h - positions[i][1])
                            new_r = min(new_r, max_r - 1e-6)
                    
                    if new_r <= 0:
                        break
                    
                    # Update all radii to new_r
                    radii[:] = new_r
                    
                    # Try to move circles to improve packing
                    # Simple gradient-free optimization: nudge circles apart
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < 2 * new_r:
                                # Push apart
                                if d > 1e-10:
                                    dir_vec = (positions[i] - positions[j]) / d
                                    push = (2 * new_r - d) / 2
                                    positions[i] += dir_vec * push
                                    positions[j] -= dir_vec * push
                    
                    # Clamp to boundaries
                    for i in range(N):
                        positions[i, 0] = np.clip(positions[i, 0], new_r, w - new_r)
                        positions[i, 1] = np.clip(positions[i, 1], new_r, h - new_r)
                
                total_sum = np.sum(radii)
                if total_sum > best_sum:
                    best_sum = total_sum
                    result = np.zeros((N, 3))
                    result[:, 0] = positions[:, 0]
                    result[:, 1] = positions[:, 1]
                    result[:, 2] = radii
                    best_result = result
    
    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
