import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145

def construct():
    t0 = time.time()
    
    # 1. Initialize Hexagonal Grid
    s = 0.0732
    dx = s
    dy = s * np.sqrt(3) / 2
    
    xs = np.arange(-0.6, 0.6, dx)
    ys = np.arange(-0.6, 0.6, dy)
    
    grid = []
    for i, y in enumerate(ys):
        offset = 0 if i % 2 == 0 else dx / 2
        for x in xs:
            grid.append((x + offset, y))
            
    pts = np.array(grid)
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    pts = pts[indices]
    pts = np.clip(pts, -0.5, 0.5)
    
    # 2. Setup
    dist_matrix = cdist(pts, pts)
    np.fill_diagonal(dist_matrix, np.inf)
    
    wall_dists = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    
    def get_global_min():
        return min(np.min(dist_matrix), np.min(wall_dists))
    
    current_d = get_global_min()
    best_d = current_d
    best_pts = pts.copy()
    
    # Optimization
    # Scale for jitter
    scale = 0.005
    
    while time.time() - t0 < _TIME_LIMIT:
        # Identify bottleneck
        min_pair_d = np.min(dist_matrix)
        min_wall_d = np.min(wall_dists)
        
        if min_wall_d < min_pair_d:
            # Wall is bottleneck
            k = np.argmin(wall_dists)
            bottleneck_type = 'wall'
        else:
            # Pair is bottleneck
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            # Pick one of them randomly
            k = i if np.random.rand() < 0.5 else j
            bottleneck_type = 'pair'
            
        # Propose move
        delta = np.random.normal(0, scale, size=2)
        new_p_k = pts[k] + delta
        new_p_k = np.clip(new_p_k, -0.5, 0.5)
        
        # Calculate new local distances
        new_dists_k = np.linalg.norm(pts - new_p_k, axis=1)
        new_dists_k[k] = np.inf
        
        new_wall_k = np.min([
            new_p_k[0] + 0.5,
            0.5 - new_p_k[0],
            new_p_k[1] + 0.5,
            0.5 - new_p_k[1]
        ])
        
        local_min_new = min(np.min(new_dists_k), new_wall_k)
        
        # Quick reject
        if local_min_new < current_d:
            continue
            
        # Calculate rest_min
        # We need min of dist_matrix excluding k, and wall_dists excluding k
        # Since N is small, just use np.min on deleted arrays
        # Or simpler: 
        # rest_min_pair = min of dist_matrix where i!=k and j!=k
        # rest_min_wall = min of wall_dists where i!=k
        
        # To avoid np.delete overhead, we can just take the global min and check if it was k?
        # No, global min is current_d.
        # If current_d was determined by k, then rest_min > current_d.
        # If current_d was determined by others, then rest_min == current_d.
        
        # Let's just compute rest_min properly.
        # dist_matrix is symmetric.
        # We can mask out row/col k.
        mask = np.ones((N, N), dtype=bool)
        mask[k, :] = False
        mask[:, k] = False
        rest_min_pair = np.min(dist_matrix[mask])
        
        mask_w = np.ones(N, dtype=bool)
        mask_w[k] = False
        rest_min_wall = np.min(wall_dists[mask_w])
        
        rest_min = min(rest_min_pair, rest_min_wall)
        
        new_global_min = min(local_min_new, rest_min)
        
        if new_global_min > current_d:
            # Accept
            dist_matrix[k, :] = new_dists_k
            dist_matrix[:, k] = new_dists_k
            wall_dists[k] = new_wall_k
            pts[k] = new_p_k
            current_d = new_global_min
            
            if current_d > best_d:
                best_d = current_d
                best_pts = pts.copy()
                
    return best_pts
