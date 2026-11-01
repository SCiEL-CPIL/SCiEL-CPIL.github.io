import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Compute min triangle area / convex hull area.
    Points: (N, 2) array.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Indices for all combinations of 3 points
    # Use pre-computed indices for speed? 
    # N=14, 364 triangles. Vectorized is fast enough.
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def neg_score_full(points):
    return -compute_score(points)

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Symmetry breaking:
    # Fix P0 = (0, 0)
    # Fix P1 = (1, 0)
    # This fixes translation and scale.
    # Rotation is still possible, but we can constrain P2 to have y > 0.
    # Actually, just fixing P0 and P1 is enough to reduce dimensions.
    # The optimizer will find the best rotation implicitly or we can just let it search.
    # But the landscape has rotational symmetry, which can cause issues.
    # Let's fix P0=(0,0) and P1=(1,0).
    # Variables: P2...P13 (12 points * 2 coords = 24 variables).
    
    def objective(x):
        # x contains coords for P2 to P13
        p2_13 = x.reshape((N-2, 2))
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        points = np.vstack((p0, p1, p2_13))
        
        # Check for degenerate points (very close together)
        # If any two points are too close, penalty
        # Compute pairwise distances
        # This is O(N^2), might be slow if done every call.
        # But 14 points is small.
        # Let's just rely on the score being 0 if degenerate.
        
        score = compute_score(points)
        return -score

    # Bounds for P2...P13
    # Since the hull area is normalized by the hull itself, the scale doesn't matter.
    # However, we fixed P1=(1,0), so the "unit" is defined by P0-P1.
    # The other points can be anywhere.
    # A reasonable bound is [-10, 10] to allow for various shapes.
    bounds = [(-10.0, 10.0)] * (2 * (N - 2))
    
    best_x = None
    best_score = -float('inf')
    
    # Run Differential Evolution
    # popsize: 10-15 is usually good
    # maxiter: limit by time
    # tol: 1e-8
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=200, # Limit iterations
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True, # Uses Nelder-Mead at the end
            updating='deferred' # Faster
        )
        
        if res.success or res.fun < best_score:
            best_score = -res.fun
            best_x = res.x
            
        # Check time
        if time.time() - start_time > deadline:
            # Reconstruct points
            p2_13 = best_x.reshape((N-2, 2))
            points = np.vstack((np.array([0.0, 0.0]), np.array([1.0, 0.0]), p2_13))
            return points
            
    except Exception as e:
        pass

    # If DE didn't finish or failed, try a few random restarts with local optimization
    # Or just use the best found so far.
    
    # Reconstruct points from best_x
    if best_x is not None:
        p2_13 = best_x.reshape((N-2, 2))
        points = np.vstack((np.array([0.0, 0.0]), np.array([1.0, 0.0]), p2_13))
        
        # Final polish with Powell on the full 28-dim space?
        # Or just on the 24-dim space?
        # Let's polish on the 24-dim space using the objective function.
        if time.time() - start_time < deadline - 10:
            try:
                res_polish = minimize(
                    objective, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                if res_polish.fun < best_score:
                    best_score = -res_polish.fun
                    best_x = res_polish.x
                    p2_13 = best_x.reshape((N-2, 2))
                    points = np.vstack((np.array([0.0, 0.0]), np.array([1.0, 0.0]), p2_13))
            except Exception:
                pass
                
        return points
    else:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        return p_reg
