import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Objective function
    def objective(params):
        # params: shape (28,) for 14 points, or (32,) for all 16 if not fixing
        # Let's fix P0=(0,0), P1=(1,0) to remove translation/rotation/scale ambiguity?
        # Actually, fixing P1=(1,0) fixes scale. 
        # If the optimal solution has d_min != 1, scaling it to d_min=1 changes the diameter.
        # The ratio is scale invariant. 
        # So we can fix P0=(0,0) and P1=(1,0).
        # If the optimizer wants to change the distance between P0 and P1, it can't.
        # This might be a constraint if the optimal pair is not P0-P1.
        # Better: Optimize all 32 coords, but subtract centroid and normalize?
        # No, simpler: Optimize all 32 coords.
        # To avoid trivial collapse, add a small penalty if any dist < epsilon?
        # Or just let the ratio go to infinity.
        
        pts = params.reshape(N, 2)
        
        # Calculate pairwise distances squared
        # Using broadcasting for speed
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        
        # Faster:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        
        # Mask diagonal
        np.fill_diagonal(dist_sq, 1e10) # Large value for max, but min needs care
        
        d_min_sq = np.min(dist_sq)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq < 1e-10:
            return 1e10
        
        return d_max_sq / d_min_sq

    # Initial guesses
    candidates = []
    
    # 1. 4x4 Grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    # Center it
    grid -= np.mean(grid, axis=0)
    candidates.append(grid.flatten())
    
    # 2. Hexagonal lattice subset
    # Generate triangular lattice points
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append((x, y))
    pts = np.array(pts)
    # Find 16 points closest to center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)[:16]
    hex_grid = pts[indices]
    hex_grid -= np.mean(hex_grid, axis=0)
    candidates.append(hex_grid.flatten())
    
    # 3. Random points in a circle
    for _ in range(5):
        r = np.sqrt(np.random.rand(N)) * 2.0
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        pts -= np.mean(pts, axis=0)
        candidates.append(pts.flatten())
        
    best_score = 1e10
    best_params = None
    
    # Optimize each candidate
    for p0 in candidates:
        if time.time() > deadline - 10:
            break
            
        # Use Nelder-Mead
        # maxiter needs to be high enough
        try:
            res = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # If we have time, polish the best solution with a tighter tolerance
    if best_params is not None and time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Return the points
    if best_params is None:
        # Fallback to grid
        return np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        
    return best_params.reshape(N, 2)
