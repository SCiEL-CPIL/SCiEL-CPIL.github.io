import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def compute_score(points):
    """
    Computes the score: min triangle area / convex hull area.
    Returns 0 if hull is degenerate.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    n = points.shape[0]
    min_area = float('inf')
    
    # Vectorized calculation of triangle areas
    # Number of triples is 286, which is small enough for a loop if vectorization is complex, 
    # but let's try vectorization for speed.
    
    # Get all pairs of indices
    # Actually, for 13 points, 286 triangles is tiny. A simple loop is fine and less memory intensive.
    # But let's use a slightly optimized loop or vectorization.
    
    # Vectorized approach:
    # Create meshgrid of indices
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    # Filter i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Define initial guesses
    def regular_polygon(n, radius=1.0, center=(0,0)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.column_stack((x, y))
    
    guesses = []
    
    # Guess 1: Regular 13-gon
    guesses.append(regular_polygon(13))
    
    # Guess 2: Regular 12-gon + Center
    pts = list(regular_polygon(12))
    pts.append([0, 0])
    guesses.append(np.array(pts))
    
    # Guess 3: Regular 11-gon + 2 points on axis
    pts = list(regular_polygon(11))
    pts.append([0.5, 0])
    pts.append([-0.5, 0])
    guesses.append(np.array(pts))
    
    # Guess 4: Regular 10-gon + 3 points in triangle
    pts = list(regular_polygon(10))
    inner_tri = regular_polygon(3, radius=0.5)
    pts.extend(inner_tri.tolist())
    guesses.append(np.array(pts))
    
    # Guess 5: Two rings (7 outer, 6 inner)
    outer = regular_polygon(7)
    inner = regular_polygon(6, radius=0.5)
    guesses.append(np.vstack((outer, inner)))
    
    # Guess 6: Random
    rng = np.random.default_rng(42)
    guesses.append(rng.random((N, 2)))
    
    best_score = 0
    best_points = None
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        # Normalize initial guess to have hull area ~ 1 to help optimization?
        # The score is scale-invariant, so we can just optimize.
        # However, Nelder-Mead works better if the scale is reasonable.
        # Let's try to scale the guess so that the hull area is 1.
        try:
            hull = ConvexHull(guess)
            area = hull.volume
            if area > 1e-6:
                scale = 1.0 / np.sqrt(area)
                guess = guess * scale
        except:
            pass
            
        # Objective function to minimize (negative score)
        def objective(x):
            pts = x.reshape(N, 2)
            return -compute_score(pts)
            
        # Run Nelder-Mead
        res = minimize(objective, guess.reshape(-1), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.success:
            final_pts = res.x.reshape(N, 2)
            final_score = -res.fun
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
                
    # If no optimization succeeded or improved, return the best guess directly
    if best_points is None:
        # Evaluate all guesses directly
        for guess in guesses:
            s = compute_score(guess)
            if s > best_score:
                best_score = s
                best_points = guess
    else:
        # Ensure best_points is actually the best among what we found
        # (It should be, but let's double check against initial guesses just in case)
        for guess in guesses:
            s = compute_score(guess)
            if s > best_score:
                best_score = s
                best_points = guess
                
    return best_points
