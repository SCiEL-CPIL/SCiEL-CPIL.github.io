import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

def compute_score(points):
    """
    Calculate the squared ratio of max to min pairwise distance.
    points: (N, D) array
    """
    # Calculate pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # Mask the diagonal (distance to self is 0)
    mask = ~np.eye(len(points), dtype=bool)
    dist_sq = dist_sq[mask]
    
    if len(dist_sq) == 0:
        return np.inf
        
    d_min_sq = np.min(dist_sq)
    d_max_sq = np.max(dist_sq)
    
    if d_min_sq < 1e-12:
        return np.inf
        
    return d_max_sq / d_min_sq

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal Lattice (4x4 block)
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    inits.append(np.array(hex_pts, dtype=float))
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. Concentric Rings (1 + 6 + 9)
    pts = [[0.0, 0.0]]
    for k in range(6):
        a = 2 * np.pi * k / 6
        pts.append([np.cos(a), np.sin(a)])
    for k in range(9):
        a = 2 * np.pi * k / 9
        pts.append([1.5 * np.cos(a), 1.5 * np.sin(a)])
    inits.append(np.array(pts, dtype=float))
    
    # 4. Random Inits
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.randn(N, D))
        
    # Optimization Loop
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Flatten for optimizer
        x0 = init.flatten()
        
        def objective(x):
            pts = x.reshape(N, D)
            return compute_score(pts)
            
        # Use Nelder-Mead
        # We can try to normalize the points to prevent drift, 
        # but Nelder-Mead is generally robust to scale invariance if initialized well.
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, D)
        except Exception:
            continue
            
        # Optional: Refine with a second pass if time permits
        # If we have time, we can do a finer search around the best found so far
        if time.time() > deadline - 10.0:
            break

    # Fallback if optimization failed
    if best_points is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points
