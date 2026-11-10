# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    def objective(flat_points):
        pts = flat_points.reshape(n, d)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)
        
    def get_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    best_points = None
    best_score = -np.inf

    # Generate initial configurations
    configs = []
    
    # 1. 4x4 Grid
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    X, Y = np.meshgrid(x, y)
    grid_pts = np.column_stack((X.ravel(), Y.ravel()))
    grid_pts -= grid_pts.mean(axis=0)
    configs.append(grid_pts)
    
    # 2. Concentric Rings: 1 + 5 + 10
    # Center
    p0 = np.array([0.0, 0.0])
    # Ring 1: 5 points, radius 1
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r1 = 1.0
    ring1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    # Ring 2: 10 points, radius 2
    angles2 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r2 = 2.0
    ring2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    ring_pts = np.vstack([p0, ring1, ring2])
    ring_pts -= ring_pts.mean(axis=0)
    configs.append(ring_pts)
    
    # 3. Concentric Rings: 1 + 6 + 9 (adjust radii)
    # Center
    p0 = np.array([0.0, 0.0])
    # Ring 1: 6 points, radius 1
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r1 = 1.0
    ring1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    # Ring 2: 9 points, radius 1.8
    angles2 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    r2 = 1.8
    ring2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    ring_pts2 = np.vstack([p0, ring1, ring2])
    ring_pts2 -= ring_pts2.mean(axis=0)
    configs.append(ring_pts2)
    
    # 4. Random with seed
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    rand_pts -= rand_pts.mean(axis=0)
    configs.append(rand_pts)
    
    # 5. Random with seed 123
    np.random.seed(123)
    rand_pts2 = np.random.randn(n, d)
    rand_pts2 -= rand_pts2.mean(axis=0)
    configs.append(rand_pts2)
    
    # 6. Hexagonal lattice subset
    # Generate a larger hex lattice and pick 16 closest to center
    hex_pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Axial to Cartesian
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= hex_pts.mean(axis=0)
    dists_to_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists_to_center)[:n]
    hex_pts_subset = hex_pts[idx]
    configs.append(hex_pts_subset)

    # Optimize each config
    for i, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Flatten for minimize
        x0 = init_pts.flatten()
        
        try:
            # Use Nelder-Mead for robustness against non-smoothness
            # maxiter needs to be high enough to converge
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            opt_pts = res.x.reshape(n, d)
            score = get_score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
                
        except Exception as e:
            continue

    # If no optimization succeeded or time ran out, use the best initial config
    if best_points is None:
        # Fallback to grid or best initial
        for init_pts in configs:
            score = get_score(init_pts)
            if score > best_score:
                best_score = score
                best_points = init_pts.copy()
        if best_points is None:
            best_points = configs[0]

    return best_points

# EVOLVE-BLOCK-END
