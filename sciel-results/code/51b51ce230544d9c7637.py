"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def construct():
    best_r = 0
    best_positions = None
    best_w = 1.0
    best_h = 1.0
    
    # Search aspect ratios
    w_vals = np.linspace(0.5, 1.5, 11)
    
    # Grid resolutions to try
    grid_sizes = [20, 30]
    
    for w in w_vals:
        h = 2.0 - w
        
        for grid_size in grid_sizes:
            # Generate grid points
            xs = np.linspace(0, w, grid_size)
            ys = np.linspace(0, h, grid_size)
            XX, YY = np.meshgrid(xs, ys)
            points = np.column_stack((XX.ravel(), YY.ravel()))
            
            # Calculate distance to boundary
            d_b = np.min(np.column_stack((points[:, 0], w - points[:, 0], 
                                          points[:, 1], h - points[:, 1])), axis=1)
            
            # Use Hungarian algorithm to assign N circles to N grid points
            # to maximize sum of d_b (minimize -d_b)
            C = -d_b.reshape(1, -1)
            C = np.repeat(C, N, axis=0)
            
            row_ind, col_ind = linear_sum_assignment(C)
            positions = points[col_ind].copy()
            
            # Add small random jitter to help escape local minima
            positions += np.random.normal(0, 0.01, positions.shape)
            positions[:, 0] = np.clip(positions[:, 0], 0, w)
            positions[:, 1] = np.clip(positions[:, 1], 0, h)
            
            # Local Search: Force-Directed
            for _ in range(150):
                # Calculate current r
                d_b_curr = np.min(np.column_stack((positions[:, 0], w - positions[:, 0], 
                                                   positions[:, 1], h - positions[:, 1])), axis=1)
                min_d_b = np.min(d_b_curr)
                
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                min_d_c = np.min(dists)
                
                r_curr = min(min_d_b, min_d_c / 2)
                
                if r_curr > best_r:
                    best_r = r_curr
                    best_positions = positions.copy()
                    best_w = w
                    best_h = h
                
                if r_curr < 1e-6:
                    break
                
                # Compute forces
                forces = np.zeros_like(positions)
                
                # Boundary forces
                for i in range(N):
                    x, y = positions[i]
                    if x < r_curr:
                        forces[i, 0] += (r_curr - x)
                    if w - x < r_curr:
                        forces[i, 0] -= (w - x - r_curr)
                    if y < r_curr:
                        forces[i, 1] += (r_curr - y)
                    if h - y < r_curr:
                        forces[i, 1] -= (h - y - r_curr)
                
                # Pairwise forces
                for i in range(N):
                    for j in range(i + 1, N):
                        d = dists[i, j]
                        if d < 2 * r_curr:
                            vec = positions[i] - positions[j]
                            norm = np.linalg.norm(vec)
                            if norm > 1e-6:
                                vec = vec / norm
                                f_mag = (2 * r_curr - d) * 0.1
                                forces[i] += vec * f_mag
                                forces[j] -= vec * f_mag
                
                # Update positions
                step_size = 0.01
                positions += forces * step_size
                positions[:, 0] = np.clip(positions[:, 0], 0, w)
                positions[:, 1] = np.clip(positions[:, 1], 0, h)
                
                if np.max(np.abs(forces)) < 1e-5:
                    break
                    
    if best_positions is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_r
    
    return result
# EVOLVE-BLOCK-END
