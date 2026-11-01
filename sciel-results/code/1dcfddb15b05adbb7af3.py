import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    # Check bounds
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    # Pairwise distances
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def get_fcc_init():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    # 3x3x3 cells of size 1/3 in [0,1]
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                # FCC basis in unit cell
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    # Shift to [-0.5, 0.5]
    fcc_pts -= 0.5
    
    # Filter points inside the cube
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    # If we have more than N, select N points with largest minimum distance
    # Greedy selection: pick point, remove neighbors, repeat
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            # Pick the first one (or random one)
            # To be deterministic, pick index 0
            pt = remaining[0]
            selected.append(pt)
            # Remove points too close to pt (within 0.15)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.15]
        fcc_pts = np.array(selected)
        
    # If we have less than N, pad with random points
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts

def construct():
    best_pts = None
    best_r = -np.inf
    
    # 1. Grid Init
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Take 50
    grid_init = grid_pts[:N].copy()
    # Add small jitter to break symmetry
    grid_init += np.random.normal(0, 0.005, size=(N, 3))
    grid_init = np.clip(grid_init, -0.5, 0.5)
    
    # 2. FCC Init
    fcc_init = get_fcc_init()
    
    # 3. Random Init
    rand_init = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    candidates = [grid_init, fcc_init, rand_init]
    
    # If time allows, add more random restarts
    while time.time() - START_TIME < TIME_LIMIT - 20.0:
        candidates.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
        
    for i, init_pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
            
        # Ensure valid start
        init_pts = np.clip(init_pts, -0.5, 0.5)
        
        # Optimize
        try:
            # Nelder-Mead
            res = minimize(neg_radius, init_pts, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            r = -res.fun
            if r > best_r:
                best_r = r
                best_pts = res.x.copy()
                
        except Exception:
            pass

    # If no valid points found, return grid
    if best_pts is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final polish: Run a few more iterations on the best point found
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res = minimize(neg_radius, best_pts, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            r = -res.fun
            if r > best_r:
                best_r = r
                best_pts = res.x.copy()
        except Exception:
            pass

    return best_pts
