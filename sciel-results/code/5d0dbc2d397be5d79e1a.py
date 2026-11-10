# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600.0  # 30s safety margin

    def objective(x):
        points = x.reshape(n, d)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-12 or d_max < 1e-12:
            return 1e10
        # Minimize log(d_max) - log(d_min) which is -log(d_min/d_max)
        return np.log(d_max) - np.log(d_min)

    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    best_score = -1.0
    best_points = np.random.randn(n, d)

    # Generate diverse starting points
    starts = []
    
    # 1. Random starts
    np.random.seed(42)
    for _ in range(10):
        starts.append(np.random.randn(n, d))
        
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. 4x4 Grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    starts.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 4. Hexagonal cluster approximation
    # Generate hex lattice points
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x_h = i + 0.5 * j
            y_h = np.sqrt(3) / 2 * j
            hex_points.append([x_h, y_h])
    hex_points = np.array(hex_points)
    # Sort by distance from center
    dists_center = np.sum(hex_points**2, axis=1)
    indices = np.argsort(dists_center)
    starts.append(hex_points[indices[:n]])
    
    # 5. Concentric rings (1 center, 7 inner, 8 outer)
    p_center = [[0, 0]]
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p_inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    p_outer = np.column_stack([1.5 * np.cos(angles_outer), 1.5 * np.sin(angles_outer)])
    starts.append(np.vstack([p_center, p_inner, p_outer]))

    # 6. Two rings of 8
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    p2 = np.column_stack([1.8 * np.cos(angles2), 1.8 * np.sin(angles2)])
    starts.append(np.vstack([p1, p2]))

    for i, start in enumerate(starts):
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(objective, start.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            opt_points = res.x.reshape(n, d)
            
            # Normalize to unit max distance for stable score calc? 
            # Score is scale invariant, so we can just calculate it.
            s = score(opt_points)
            if s > best_score:
                best_score = s
                best_points = opt_points
        except Exception:
            pass

    # Final check and return
    if time.time() > deadline - 10:
        return best_points

    return best_points

# EVOLVE-BLOCK-END
