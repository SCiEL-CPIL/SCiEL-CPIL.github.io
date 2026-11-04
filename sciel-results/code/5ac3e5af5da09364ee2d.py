import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def _triangle_areas(points):
    n = points.shape[0]
    indices = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _objective_reduced(params):
    """
    params: 24 values for points 2..12.
    P0 = [0,0], P1 = [1,0].
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    rest = params.reshape((N-2, 2))
    points = np.vstack([p0, p1, rest])
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 1e10
        
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    score = min_area / hull_area
    return -score

def _regular_polygon_reduced(n):
    """
    Generate regular n-gon, rotate/translate/scale so that P0=[0,0] and P1=[1,0].
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Shift so that one vertex is at angle 0?
    # Regular polygon vertices on unit circle.
    # We need P0 and P1 to be specific points.
    # Let's just generate a regular polygon and then transform it.
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # We want to map pts[0] to (0,0) and pts[1] to (1,0).
    # This is an affine transformation.
    # v0 = pts[0], v1 = pts[1].
    # Target: t0 = (0,0), t1 = (1,0).
    # T(x) = A(x - v0) + t0.
    # T(v1) = A(v1 - v0) = (1,0).
    # Let d = v1 - v0.
    # A d = (1,0).
    # We can choose A to be the rotation+scale that maps d to (1,0).
    # d = [dx, dy].
    # We want A [dx, dy]^T = [1, 0]^T.
    # Let A = [[a, b], [c, d]].
    # a dx + b dy = 1
    # c dx + d dy = 0
    # Simple solution:
    # Scale by 1/|d| and rotate by -angle(d).
    
    d = pts[1] - pts[0]
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-9:
        return np.zeros((n-2, 2)) # Degenerate
        
    # Rotation matrix that maps d to [1,0]
    # d = norm_d * [cos(theta), sin(theta)]
    # We want R * d = [1,0]
    # R = [[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]] * (1/norm_d)?
    # Actually, just compute the transformation.
    
    # Vectorized transform for all points
    shifted = pts - pts[0]
    
    # Calculate rotation
    theta = np.arctan2(d[1], d[0])
    c = np.cos(-theta)
    s = np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    
    # Scale
    S = 1.0 / norm_d
    
    # Apply
    transformed = (R @ shifted.T).T * S
    
    # Return points 2 to n-1
    return transformed[2:]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_score = -np.inf
    best_points = None
    
    # Initializations in reduced space
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon_reduced(N))
    
    # 2. Random points in a box, transformed to reduced space?
    # Easier: Generate full random points, then transform to reduced space.
    # Or just generate random points for P2..P12 in a reasonable range, e.g., [-2, 2].
    for _ in range(10):
        # Random points
        # To be safe, start them in a circle around the origin?
        # P0=(0,0), P1=(1,0).
        # Other points should be in a similar scale.
        inits.append(np.random.rand(N-2, 2) * 2 - 1)
        
    # 3. Grid-like
    # Just random is probably fine for 10 restarts.

    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        options = {
            'maxiter': 1000,
            'xatol': 1e-5,
            'fatol': 1e-7,
            'disp': False
        }
        
        try:
            res = minimize(_objective_reduced, x0, method='Nelder-Mead', options=options)
            if res.fun < best_score:
                # Reconstruct full points
                p0 = np.array([0.0, 0.0])
                p1 = np.array([1.0, 0.0])
                rest = res.x.reshape((N-2, 2))
                points = np.vstack([p0, p1, rest])
                
                hull_area = _hull_area(points)
                if hull_area > 1e-9:
                    areas = _triangle_areas(points)
                    min_area = np.min(areas)
                    score = min_area / hull_area
                    if score > best_score:
                        best_score = score
                        best_points = points
        except Exception:
            pass

    if best_points is None:
        # Fallback
        full_init = _regular_polygon_reduced(N)
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        best_points = np.vstack([p0, p1, full_init])
        
    return best_points
