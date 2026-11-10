import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 50

def _pairwise_distances(pos):
    """Compute all pairwise distances efficiently."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def _compute_radius_fast(positions):
    """Fast radius computation using vectorized operations."""
    # Check bounds first
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    dists = _pairwise_distances(positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_bound, min_dist / 2.0)

def compute_radius(positions):
    """
    Compute the maximum common radius for given circle centers.
    positions: array of shape (N, 2)
    """
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def project_to_box(pos):
    """Project positions into the unit square [-0.5, 0.5]^2."""
    return np.clip(pos, -0.5, 0.5)

def generate_grid_init(seed=0):
    """Generate a hexagonal grid initialization with jitter."""
    np.random.seed(seed)
    positions = []
    sqrt3 = np.sqrt(3.0)
    
    # Try different grid configurations
    best_pos = None
    best_r = 0.0

    # Hexagonal packing approximation
    # 50 circles.
    # Rows: 5, 6, 7, 8, 9?
    # 5 rows: 10 cols. 5*10=50.
    # 6 rows: 8-9 cols.
    
    for n_rows in range(5, 10):
        n_cols = int(np.ceil(N / n_rows))
        
        # Calculate spacing
        # Width = 1.0, Height = 1.0
        # s_x = 1.0 / n_cols
        # s_y = s_x * sqrt(3) / 2
        # Total height = (n_rows - 1) * s_y + s_x (diameter approx)
        # We want to fit in 1.0.
        
        s_x = 1.0 / n_cols
        s_y = s_x * sqrt3 / 2.0
        
        # Check if it fits in height
        # If not, scale down
        total_h = (n_rows - 1) * s_y + s_x
        if total_h > 1.0:
             scale = 1.0 / total_h
             s_x *= scale
             s_y *= scale
        
        pos_list = []
        for row in range(n_rows):
            y = -0.5 + s_x/2 + row * s_y
            # Offset every other row
            if row % 2 == 1:
                x_start = -0.5 + s_x
            else:
                x_start = -0.5 + s_x/2
            
            # Determine number of points in this row
            # We need exactly N points total
            # Simple approach: fill row by row
            for col in range(n_cols):
                x = x_start + col * s_x
                pos_list.append([x, y])
            if len(pos_list) >= N:
                break
        
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            # Jitter
            pos_arr += np.random.uniform(-0.01, 0.01, pos_arr.shape)
            pos_arr = project_to_box(pos_arr)
            r = _compute_radius_fast(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
                
    # Fallback: Simple square grid
    if best_pos is None or best_r < 0.05:
         n_grid = 7 # 7x7=49, add 1? or 5x10
         # Let's just use random if grid fails
         pos_arr = np.random.uniform(-0.4, 0.4, (N, 2))
         best_pos = pos_arr
         best_r = _compute_radius_fast(best_pos)

    return best_pos

def nm_optimize(positions, time_limit=10.0, seed=0):
    """
    Optimize using Nelder-Mead on the negative exact radius.
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = _compute_radius_fast(best_pos)
    
    def objective(x):
        pos = x.reshape(N, 2)
        # Project to box to maintain feasibility
        pos = project_to_box(pos)
        r = _compute_radius_fast(pos)
        return -r

    # Nelder-Mead settings
    # Initial simplex size
    # We want to explore locally
    init_size = 0.01
    
    try:
        result = minimize(
            objective, 
            best_pos.flatten(), 
            method='Nelder-Mead',
            options={
                'xatol': 1e-6,
                'fatol': 1e-6,
                'maxiter': 2000,
                'maxfev': 5000,
                'initial_simplex_size': init_size,
                'adaptive': True
            }
        )
        
        new_pos = project_to_box(result.x.reshape(N, 2))
        new_r = _compute_radius_fast(new_pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos.copy()
            
    except Exception:
        pass
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_pos = None
    best_r = 0.0

    # 1. Generate initial candidates
    # Try a few different seeds for grid init
    candidates = []
    for i in range(5):
        pos = generate_grid_init(seed=i)
        candidates.append(pos)
        
    # Also add a random start
    candidates.append(np.random.uniform(-0.4, 0.4, (N, 2)))
    
    # 2. Optimize each candidate
    # Allocate time based on number of candidates
    n_candidates = len(candidates)
    time_per_candidate = min(20.0, (deadline - 10.0) / n_candidates)
    
    for i, pos_init in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
            
        current_time = time.time()
        remaining_time = deadline - 5.0 - current_time
        alloc_time = min(time_per_candidate, remaining_time)
        
        if alloc_time < 1.0:
            break
            
        pos_opt, r_opt = nm_optimize(pos_init, time_limit=alloc_time, seed=100+i)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pos = pos_opt.copy()
            
    # 3. Final Polish on the best result
    if best_pos is not None and time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        if remaining > 1.0:
            # Run NM again with tighter tolerance on the best result
            pos_final, r_final = nm_optimize(best_pos, time_limit=remaining, seed=999)
            if r_final > best_r:
                best_r = r_final
                best_pos = pos_final.copy()
                
    # Ensure valid output
    if best_pos is None:
        best_pos = generate_grid_init(seed=0)
        
    best_pos = project_to_box(best_pos)
    
    return best_pos

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = compute_radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.10f}")
