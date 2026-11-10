import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _calc_score(points):
    """
    Calculate the score: A_min / A_hull.
    Returns 0 if invalid.
    """
    if points.shape != (13, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check for collinearity / hull validity
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Calculate min triangle area
    n = 13
    min_area = np.inf
    # Vectorized approach for speed, though 286 is small
    # We can use broadcasting for all triples?
    # i, j, k indices
    # It's faster to just loop in Python for 286 items than to create huge arrays
    # But let's try a semi-vectorized approach or just simple loop.
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            # Vectorize over k
            # k > j
            if j + 1 >= n:
                continue
            pk = points[j+1:]
            # Area = 0.5 * | (pj - pi) x (pk - pi) |
            # Cross product in 2D: (x1*y2 - y1*x2)
            v1 = pj - pi
            v2 = pk - pi
            # v1 is (2,), v2 is (m, 2)
            # cross = v1[0]*v2[:,1] - v1[1]*v2[:,0]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
                if min_area < 1e-10:
                    return 0.0 # Collinear or very close

    if min_area == np.inf:
        return 0.0
        
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for minimizer.
    We minimize the negative score.
    """
    points = points_flat.reshape(13, 2)
    score = _calc_score(points)
    if score == 0.0:
        return 1000.0 # Large penalty
    return -score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0 # 20s buffer
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    candidates = []
    
    # Guess 1: 4x4 grid, remove 3 corners
    # Grid points in [-1, 1]
    coords = np.array([-1, -1/3, 1/3, 1])
    grid = np.array([[x, y] for x in coords for y in coords])
    # Indices of corners: 0, 3, 12, 15
    # Remove 3 corners: 0, 3, 15
    mask = np.ones(16, dtype=bool)
    mask[[0, 3, 15]] = False
    candidates.append(grid[mask])
    
    # Guess 2: 4x4 grid, remove 3 random points (seeded)
    rng = np.random.default_rng(42)
    idx = rng.choice(16, size=3, replace=False)
    mask2 = np.ones(16, dtype=bool)
    mask2[idx] = False
    candidates.append(grid[mask2])
    
    # Guess 3: 3x5 grid, remove 2 points
    coords3 = np.array([-1, 0, 1])
    coords5 = np.array([-1, -0.5, 0, 0.5, 1])
    grid35 = np.array([[x, y] for x in coords3 for y in coords5])
    # Remove 2 corners
    # Corners of 3x5: (0,0), (0,4), (2,0), (2,4)
    # Indices: 0, 4, 10, 14
    mask3 = np.ones(15, dtype=bool)
    mask3[[0, 14]] = False
    candidates.append(grid35[mask3])
    
    # Guess 4: Regular 13-gon (might be bad, but good start for some optimizers)
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Guess 5: Random points in square
    candidates.append(rng.random((13, 2)) * 2 - 1)
    
    # Guess 6: Jittered 4x4 grid (remove 3 corners)
    jitter = rng.random((13, 2)) * 0.2 - 0.1
    candidates.append(candidates[0] + jitter)
    
    for init_pts in candidates:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            res = minimize(
                _objective, 
                init_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            opt_pts = res.x.reshape(13, 2)
            score = _calc_score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts
                
        except Exception:
            pass

    # If no optimization succeeded or time ran out, try a quick local search on best candidate
    # Or just return best found.
    
    if best_points is None:
        # Fallback: return the best initial candidate
        for init_pts in candidates:
            s = _calc_score(init_pts)
            if s > best_score:
                best_score = s
                best_points = init_pts
                
    return best_points
