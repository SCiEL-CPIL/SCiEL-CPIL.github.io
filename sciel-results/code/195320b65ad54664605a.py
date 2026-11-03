import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_radii(pos, W, H):
    """
    Compute the maximum radius for each circle given positions pos in a W x H rectangle.
    pos: (N, 2) array
    """
    n = len(pos)
    # Distances to walls
    dist_walls = np.minimum(pos, np.array([W, H]) - pos)
    # The distance to the closest wall for each circle
    wall_dist = np.min(dist_walls, axis=1)
    
    # Pairwise distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    
    # The closest other circle for each i is min_j dists[i, j]
    # The max radius is half of that distance
    min_circle_dist = np.min(dists, axis=1) / 2.0
    
    # Radius is min of wall distance and half closest circle distance
    radii = np.minimum(wall_dist, min_circle_dist)
    
    # Ensure radii are non-negative (should be if pos is in [0,W]x[0,H])
    radii = np.maximum(radii, 0)
    
    return radii

def _pack_for_aspect(W, H):
    """
    Pack N circles in W x H rectangle to maximize sum of radii.
    Returns (positions, radii).
    """
    # Initialization: Staggered Grid
    # Determine grid size
    # Approximate number of circles per row/col
    aspect = W / H
    if aspect >= 1:
        cols = int(np.round(np.sqrt(N * aspect)))
    else:
        cols = int(np.round(np.sqrt(N / aspect)))
    if cols < 1: cols = 1
    rows = int(np.ceil(N / cols))
    
    # If rows * cols < N, increase cols
    while rows * cols < N:
        cols += 1
        rows = int(np.ceil(N / cols))
        
    positions = np.zeros((N, 2))
    for i in range(N):
        col = i % cols
        row = i // cols
        
        # Grid spacing
        dx = W / cols
        dy = H / rows
        
        # Center in cell
        x = (col + 0.5) * dx
        y = (row + 0.5) * dy
        
        # Stagger odd rows
        if row % 2 == 1:
            x += dx / 2.0
            # Wrap around if it goes out of bounds? 
            # Actually, for packing, shifting might push out. 
            # Better to just shift and clamp, or use a hexagonal lattice approach.
            # Let's just shift and clamp to [dx/2, W-dx/2] roughly?
            # Simpler: just use a uniform grid first, let optimizer do the work.
            # But staggered is a better start.
            # Let's clamp x to [0, W]
            x = np.clip(x, 0, W)
            
        positions[i, 0] = x
        positions[i, 1] = y
        
    # Clamp all positions to [0, W] x [0, H]
    positions[:, 0] = np.clip(positions[:, 0], 0, W)
    positions[:, 1] = np.clip(positions[:, 1], 0, H)
    
    # Objective function
    def objective(x):
        pos = x.reshape(N, 2)
        # If any position is out of bounds, return a large penalty
        # Nelder-Mead can go out of bounds.
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > W) or \
           np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > H):
            # Soft penalty or hard reject?
            # Hard reject can cause issues. Let's use a large penalty.
            out_of_bounds = np.sum(np.maximum(0, -pos[:, 0])) + np.sum(np.maximum(0, pos[:, 0] - W)) + \
                            np.sum(np.maximum(0, -pos[:, 1])) + np.sum(np.maximum(0, pos[:, 1] - H))
            if out_of_bounds > 1e-6:
                return 1e6 + out_of_bounds * 100
        radii = _compute_radii(pos, W, H)
        return -np.sum(radii)
    
    # Optimization
    x0 = positions.flatten()
    
    # Use Nelder-Mead as it doesn't require gradients and handles the non-smooth min well
    # maxiter needs to be high enough, but we have a time budget.
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    best_pos = result.x.reshape(N, 2)
    # Clamp to be safe
    best_pos[:, 0] = np.clip(best_pos[:, 0], 0, W)
    best_pos[:, 1] = np.clip(best_pos[:, 1], 0, H)
    
    best_radii = _compute_radii(best_pos, W, H)
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_score = -np.inf
    best_result = None
    
    # Try various aspect ratios
    # Square (1.0) is a good start. Wide (2.0, 3.0) and Tall (0.5, 0.3) too.
    aspect_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]
    
    # Sort by closeness to 1.0? Or just iterate.
    # Let's prioritize square and near-square.
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        try:
            pos, radii = _pack_for_aspect(W, H)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                # Create the result array (x, y, r)
                # Note: The evaluator checks the bounding box of the returned circles.
                # Our circles are inside [0, W] x [0, H].
                # The bounding box of the circles will have width <= W and height <= H.
                # Perimeter = 2*(w_box + h_box) <= 2*(W+H) = 4.
                # So this is valid.
                best_result = np.column_stack([pos, radii])
        except Exception as e:
            # If optimization fails, skip this aspect ratio
            pass
    
    # Fallback: Simple grid if nothing worked
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        # 5x5 grid, take 21
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) < N:
                    x = (i + 0.5) / 5.0
                    y = (j + 0.5) / 5.0
                    r = 0.5 / 5.0 - 1e-3
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    return best_result
