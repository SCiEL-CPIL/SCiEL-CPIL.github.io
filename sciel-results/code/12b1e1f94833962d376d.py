"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def construct():
    # Pack 21 circles in a rectangle with perimeter 4 (w + h = 2)
    # Maximize sum of radii using grid-based approach with local optimization
    
    best_result = None
    best_sum = -1
    
    # Try different grid configurations and aspect ratios
    configs = [
        (5, 5),  # 5x5 grid
        (4, 6),  # 4x6 grid
        (6, 4),  # 6x4 grid
        (3, 7),  # 3x7 grid
        (7, 3),  # 7x3 grid
        (5, 4),  # 5x4 grid (20 cells, need 21 - skip)
        (4, 5),  # 4x5 grid (20 cells, need 21 - skip)
        (3, 8),  # 3x8 grid
        (8, 3),  # 8x3 grid
        (2, 11), # 2x11 grid
        (11, 2), # 11x2 grid
        (1, 21), # 1x21 grid
        (21, 1), # 21x1 grid
    ]
    
    # Aspect ratios to try (w_frac = w / 2, so w = 2*w_frac, h = 2 - w)
    w_fracs = np.linspace(0.25, 0.75, 21)  # 21 aspect ratios
    
    for nx, ny in configs:
        if nx * ny < N:
            continue
        
        for w_frac in w_fracs:
            w = 2 * w_frac
            h = 2 - w
            
            if w <= 0 or h <= 0:
                continue
            
            # Create grid positions with proper spacing
            # Use linspace to place circles at centers of grid cells
            if nx > 1:
                xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
            else:
                xs = np.array([w / 2])
            
            if ny > 1:
                ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
            else:
                ys = np.array([h / 2])
            
            # Generate all grid positions
            positions = []
            for i in range(nx):
                for j in range(ny):
                    positions.append([xs[i], ys[j]])
            
            positions = np.array(positions[:N])
            
            # Compute initial radius from minimum pairwise distance and boundary constraints
            if N > 1:
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                min_dists = np.min(dists, axis=1)
                r_pair = np.min(min_dists) / 2
            else:
                r_pair = min(w, h) / 2
            
            # Boundary constraint: radius limited by distance to nearest edge
            r_boundary = np.min([
                np.min(positions[:, 0]),
                np.min(w - positions[:, 0]),
                np.min(positions[:, 1]),
                np.min(h - positions[:, 1])
            ])
            
            r = min(r_pair, r_boundary) - 1e-6
            
            if r <= 0:
                continue
            
            # Local optimization: iteratively adjust positions to maximize minimum radius
            # Use a simple gradient-free approach: try small perturbations
            current_positions = positions.copy()
            current_r = r
            
            # Try to improve by local search
            for iteration in range(30):
                # Compute current minimum distance
                min_dist = np.inf
                for i in range(N):
                    # Boundary distances
                    d_b = min(current_positions[i][0], w - current_positions[i][0],
                             current_positions[i][1], h - current_positions[i][1])
                    min_dist = min(min_dist, d_b)
                    # Pairwise distances
                    for j in range(i + 1, N):
                        d = np.linalg.norm(current_positions[i] - current_positions[j])
                        min_dist = min(min_dist, d)
                
                new_r = min_dist / 2 - 1e-7
                
                if new_r <= current_r + 1e-8:
                    break
                
                current_r = new_r
                
                # Try to improve positions: move circles away from each other and boundaries
                improved = False
                for i in range(N):
                    # Compute force vector: away from other circles and toward center
                    force = np.zeros(2)
                    
                    # Repulsion from other circles
                    for j in range(N):
                        if i == j:
                            continue
                        diff = current_positions[i] - current_positions[j]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-6:
                            continue
                        # Repulsive force proportional to 1/dist^2
                        force += (diff / dist) * (1.0 / (dist * dist + 1e-6))
                    
                    # Attraction to center (weak)
                    center = np.array([w / 2, h / 2])
                    force += (center - current_positions[i]) * 0.1
                    
                    # Try small step in direction of force
                    step_size = 0.01
                    new_pos = current_positions[i] + force * step_size
                    
                    # Check if new position is valid (within bounds)
                    if (new_pos[0] >= 0 and new_pos[0] <= w and 
                        new_pos[1] >= 0 and new_pos[1] <= h):
                        # Check if this improves the minimum distance
                        old_min = min_dist
                        # Compute new minimum distance with this circle moved
                        new_min = np.inf
                        for k in range(N):
                            if k == i:
                                pos_k = new_pos
                            else:
                                pos_k = current_positions[k]
                            # Boundary distance
                            d_b = min(pos_k[0], w - pos_k[0], pos_k[1], h - pos_k[1])
                            new_min = min(new_min, d_b)
                            # Pairwise
                            for l in range(k + 1, N):
                                if l == i:
                                    continue
                                if k == i:
                                    continue
                                diff2 = current_positions[k] - current_positions[l]
                                d = np.linalg.norm(diff2)
                                new_min = min(new_min, d)
                        
                        if new_min > old_min + 1e-8:
                            current_positions[i] = new_pos
                            improved = True
                
                if not improved:
                    break
            
            total_sum = N * current_r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = current_positions[:, 0]
                result[:, 1] = current_positions[:, 1]
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
