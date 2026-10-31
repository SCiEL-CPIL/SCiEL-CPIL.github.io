import numpy as np
from scipy.optimize import minimize

def construct():
    def objective(x):
        pts = x.reshape(-1, 2)
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 0)
        
        min_d = np.min(dists)
        max_d = np.max(dists)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    best_score = float('inf')
    best_x = None
    
    n = 16
    dim = 2 * n
    
    # Generate several initial guesses
    seeds = []
    
    # 1. 4x4 Grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid.flatten())
    
    # 2. Random Uniform in Unit Circle
    np.random.seed(42)
    for _ in range(10):
        angles = np.random.rand(n) * 2 * np.pi
        radii = np.sqrt(np.random.rand(n))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts.flatten())
        
    # 3. Hexagonal Seed (Generated independently)
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = q + r/2.0
                y = np.sqrt(3)/2.0 * r
                hex_pts.append([x, y])
    if len(hex_pts) >= 16:
        seeds.append(np.array(hex_pts[:16]).flatten())
        
    # Optimize each seed
    for x0 in seeds:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to the first seed
        return np.array(seeds[0]).reshape(-1, 2)
        
    return best_x.reshape(-1, 2)
