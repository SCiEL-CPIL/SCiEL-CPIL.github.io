import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = np.inf
    # There are C(13, 3) = 286 triangles. 
    # Vectorized calculation is possible but for N=13, a loop is fast enough.
    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # If hull fails (e.g., collinear points), return a very small area
        # to penalize this configuration
        return 1e-10

def _score(pts):
    """Calculate the objective score: min_triangle_area / hull_area."""
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg_pts)
    
    # 2. Perturbed regular 13-gons
    rng = np.random.default_rng(42)
    for i in range(10):
        noise = rng.normal(0, 0.1, (N, 2))
        perturbed = reg_pts + noise
        candidates.append(perturbed)
        
    # 3. Random points on circle
    for i in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        # Sort angles to ensure convex order? Not strictly necessary for area calc,
        # but helps local search.
        angles = np.sort(angles)
        rand_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(rand_pts)
        
    # 4. Random points in square (will be pushed to hull by optimizer)
    for i in range(10):
        rand_pts = rng.uniform(-1, 1, (N, 2))
        candidates.append(rand_pts)

    # Define the objective function for minimization (we want to maximize score)
    def objective(x):
        pts = x.reshape(N, 2)
        s = _score(pts)
        # Add a small penalty for points that are not in convex position?
        # Actually, if a point is inside, the min triangle area usually drops significantly.
        # We can just minimize -s.
        return -s

    # Run optimization for each candidate
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        try:
            # Nelder-Mead is robust for this
            res = minimize(objective, init_pts.flatten(), method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000})
            
            if res.success or res.fun < 0: # fun is -score, so lower is better
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_pts
                    
        except Exception as e:
            pass

    # If no optimization succeeded or time ran out, use the best initial candidate
    if best_points is None:
        # Just pick the regular polygon as fallback
        best_points = candidates[0]
        best_score = _score(best_points)

    # Final check: ensure best_points is a valid numpy array
    return best_points

# EVOLVE-BLOCK-END
