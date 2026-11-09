# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    # points shape (n, 2)
    # We need to compute area for all combinations of 3 points.
    # Number of triangles is n*(n-1)*(n-2)/6. For n=13, this is 286.
    # We can do this without a loop for speed, or with a simple loop which is fine for 286.
    # Let's use a loop for clarity and sufficient speed, or vectorize if needed.
    # Vectorization:
    # Create indices for all pairs (i, j, k) with i < j < k
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |det([p2-p1, p3-p1])|
    # det = (p2x - p1x)*(p3y - p1y) - (p2y - p1y)*(p3x - p1x)
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn ratio.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.0
    init_regular = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    inits.append(init_regular)
    
    # 2. Concentric polygons: 7 outer, 6 inner (approx)
    # Or 1 center, 6 inner, 6 outer?
    # Let's try 1 center, 6 on circle r=0.5, 6 on circle r=1.0
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    init_concentric = np.vstack([center, inner_pts, outer_pts])
    inits.append(init_concentric)
    
    # 3. Random points in unit square, scaled
    rng = np.random.default_rng(seed=123)
    init_random = rng.random((n, 2)) * 2 - 1
    inits.append(init_random)
    
    # 4. Grid-like jitter
    # 4x4 grid minus one corner, jittered
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Remove one point to get 13? 16 points. Remove center-ish?
    # Just take first 13
    pts = pts[:13]
    # Add slight jitter
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)

    # Optimize each initial configuration
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Nelder-Mead optimization
        # We want to maximize score, so minimize -score
        # But score can be 0 if invalid. We can add a penalty.
        def objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0 # Penalty
            return -s
        
        # Initial simplex
        x0 = init_pts.flatten()
        
        # Method: Nelder-Mead is good for small dimensions (26 vars)
        # maxiter needs to be high enough
        # To save time, we limit maxiter
        max_iter = 500
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < 0: # If fun < 0, it found a valid positive score
                final_pts = res.x.reshape((n, 2))
                final_score = compute_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points = final_pts.copy()
                    
        except Exception as e:
            pass
            
        # Also try a few random perturbations of the best found so far?
        # Or just stick to the restarts.
        
    # If no optimization succeeded or score is still low, fall back to a known good static configuration
    # A regular 13-gon is a decent baseline
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    # Final check and return
    return best_points

# EVOLVE-BLOCK-END
