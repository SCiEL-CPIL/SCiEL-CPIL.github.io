"""Improved packing using hexagonal initialization and local optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0:
        return 0.0
    # Distance to walls
    wall_dist = np.min(np.abs(0.5 - np.abs(centers)))
    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N, d_scale=1.0):
    """Generate a hexagonal grid of points."""
    # Estimate d
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * d_scale
    
    points = []
    # Generate rows
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback: shrink d and retry? Or just pad with random?
        # For now, if we don't have enough, we might have a bad estimate.
        # Let's just return what we have and let the caller handle it?
        # Better: return a square grid if hex fails?
        # Let's try to generate more by reducing d
        pass 
        
    return points[:N] if len(points) >= N else None

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_centers = None
    best_score = -1.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # 2. Square Grid (Fallback/Comparison)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 3. Random Jittered Grid
    np.random.seed(42)
    jittered = grid[:N].copy()
    jittered += np.random.normal(0, 0.01, jittered.shape)
    jittered = np.clip(jittered, -0.49, 0.49)
    inits.append(jittered)
    
    # 4. Additional random initializations for diversity
    for _ in range(3):
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
    
    for init_centers in inits:
        if time.time() - start_time > time_limit:
            break
            
        x0 = init_centers.flatten()
        
        def objective(x):
            X = x.reshape(N, 2)
            D2 = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
            np.fill_diagonal(D2, 1.0)
            return np.sum(1.0 / D2)
            
        def gradient(x):
            X = x.reshape(N, 2)
            D2 = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
            np.fill_diagonal(D2, 1.0)
            M = 1.0 / (D2**2)
            np.fill_diagonal(M, 0.0)
            
            RowSumM = M.sum(axis=1)
            MX = M @ X
            
            Grad = -2.0 * (RowSumM[:, np.newaxis] * X - MX)
            return Grad.flatten()
            
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2*N),
                           options={'maxiter': 500, 'disp': False})
            final_centers = res.x.reshape(N, 2)
            score = _compute_score(final_centers)
            
            if score > best_score:
                best_score = score
                best_centers = final_centers
                
        except Exception:
            pass
            
    # If no optimization succeeded, return the best initial
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
