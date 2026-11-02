import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.special import logsumexp

N = 13
DEADLINE = 145.0

def _min_triangle_area_smooth(points, beta=50.0):
    n = len(points)
    if n < 3:
        return 0.0
    
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    val = logsumexp(-beta * areas)
    return -val / beta

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _objective(x, beta=50.0):
    pts = x.reshape(-1, 2)
    min_area_smooth = _min_triangle_area_smooth(pts, beta)
    hull_area = _convex_hull_area(pts)
    
    if hull_area < 1e-10:
        return 1e10
        
    ratio = min_area_smooth / hull_area
    return -ratio

def _true_score(points):
    n = len(points)
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10: return 0.0
    return min_area / hull_area

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    
    best_points = _regular_polygon(N, radius=1.0)
    best_score = _true_score(best_points)
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon(N, radius=1.0))
    
    # 2. Jittered regular polygons
    for _ in range(30):
        pts = _regular_polygon(N, radius=1.0)
        noise = np.random.normal(0, 0.15, pts.shape)
        pts += noise
        inits.append(pts)
        
    # 3. Random convex sets (sorted angles, random radii)
    for _ in range(20):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.random.uniform(0.3, 1.0, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))

    # Optimization Loop
    for init in inits:
        if time.time() - start_time > DEADLINE:
            break
            
        current_pts = init.copy()
        
        # Annealing: Start smooth, get sharp
        # This helps navigate the landscape
        betas = [20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE:
                break
                
            x0 = current_pts.flatten()
            try:
                res = minimize(
                    _objective,
                    x0,
                    args=(beta,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
                
                # Check if the result is better on the smooth objective
                # Note: _objective returns negative ratio
                if res.fun < _objective(x0, beta):
                    current_pts = res.x.reshape(-1, 2)
                    
            except Exception:
                continue
        
        # Evaluate true score
        score = _true_score(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts

    return best_points
