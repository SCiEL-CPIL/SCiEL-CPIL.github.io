import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def calc_score(points):
    """Calculate min triangle area / convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = float('inf')
    # Use a slightly more efficient loop structure
    for i in range(N-2):
        p1 = points[i]
        for j in range(i+1, N-1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j+1, N):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def objective(points):
    return -calc_score(points)

def regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def concentric_rings(n1, n2, r1, r2, offset=0.0):
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + offset
    p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([p1, p2])

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    best_score = -1
    best_points = None
    
    # Initial candidates
    candidates = [
        regular_polygon(N),
        concentric_rings(7, 7, 0.5, 1.0, offset=np.pi/7),
        concentric_rings(6, 8, 0.4, 1.0, offset=np.pi/8),
        concentric_rings(1, 13, 0.0, 1.0, offset=0.0),
        concentric_rings(5, 9, 0.6, 1.0, offset=np.pi/9),
    ]
    
    # Add random jitter to regular polygon
    rng = np.random.default_rng(42)
    jittered = regular_polygon(N) + rng.normal(0, 0.1, size=(N, 2))
    candidates.append(jittered)
    
    for x0 in candidates:
        if time.time() > deadline:
            break
            
        try:
            # Powell is robust for non-smooth functions
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'maxfev': 1000, 'ftol': 1e-12}
            )
            
            score = calc_score(res.x)
            if score > best_score:
                best_score = score
                best_points = res.x
        except Exception:
            continue
            
    # If we have time, refine the best candidate
    if best_points is not None and time.time() < deadline - 5:
        try:
            res2 = scipy.optimize.minimize(
                objective, 
                best_points, 
                method='Powell', 
                options={'maxiter': 200, 'maxfev': 1000, 'ftol': 1e-15}
            )
            score2 = calc_score(res2.x)
            if score2 > best_score:
                best_points = res2.x
        except Exception:
            pass
            
    if best_points is None:
        best_points = regular_polygon(N)
        
    return best_points
