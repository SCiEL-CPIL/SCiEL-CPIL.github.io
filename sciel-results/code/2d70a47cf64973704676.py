import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    # There are N*(N-1)*(N-2)/6 triangles
    n = len(points)
    min_area = float('inf')
    
    # Optimization: Use vectorized calculation if possible, but N=14 is small
    # Loop is fine: 364 triangles
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle with vertices p_i, p_j, p_k
                # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
                        
    return min_area / hull_area

def neg_score(x):
    """
    Objective function for minimization (we maximize score).
    x is a flat array of length 2*N.
    """
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Generate several initial configurations
    inits = []
    
    # 1. Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # 2. Regular N-gon with small random noise
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        inits.append(p_reg + noise)
        
    # 3. Two concentric rings (7 points each)
    # Inner ring radius 0.5, outer ring radius 1.0
    # Rotate inner ring by pi/7 to interleave
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi/7
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)))
    p_two_rings = np.vstack((p_outer, p_inner))
    inits.append(p_two_rings)
    
    # 4. Random points in unit circle
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        p_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(p_rand)

    # Evaluate initial scores
    for p in inits:
        s = compute_score(p)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            
    # Local optimization from best initial points
    # Sort inits by initial score descending to prioritize good starts
    scored_inits = [(compute_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    for init_score, init_points in scored_inits:
        if time.time() - start_time > deadline:
            break
            
        x0 = init_points.flatten()
        
        # Use Nelder-Mead
        # maxiter needs to be large enough to converge
        # We can also use 'Powell' which is often better for non-smooth
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < -init_score: # If it improved or succeeded
                new_points = res.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
            
        # Try Powell as well if time permits
        if time.time() - start_time > deadline:
            break
        try:
            res2 = minimize(neg_score, x0, method='Powell', 
                            options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
            if res2.success or res2.fun < -init_score:
                new_points = res2.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass

    # Final polish: Run a few more iterations on the best point found
    if best_points is not None and time.time() - start_time < deadline - 5:
        x_best = best_points.flatten()
        try:
            res_final = minimize(neg_score, x_best, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            new_points = res_final.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points
