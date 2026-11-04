import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14
TRIPLES = np.array(list(itertools.combinations(range(N), 3)))

def _get_areas(points):
    # Vectorized area calculation
    # points: (N, 2)
    # For each triple (i,j,k), area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    # We can compute this for all triples at once.
    p_i = points[TRIPLES[:, 0]]
    p_j = points[TRIPLES[:, 1]]
    p_k = points[TRIPLES[:, 2]]
    
    # Cross product in 2D: (x1*y2 - y1*x2)
    # det(p_j - p_i, p_k - p_i) = (p_j[0]-p_i[0])*(p_k[1]-p_i[1]) - (p_j[1]-p_i[1])*(p_k[0]-p_i[0])
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    dets = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(dets)

def _get_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    areas = _get_areas(points)
    hull_area = _get_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_score(points, temp=0.05):
    if points.shape != (N, 2):
        return 0.0
    areas = _get_areas(points)
    hull_area = _get_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    
    # Soft min of areas
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temp))
    soft_min = m - temp * np.log(s)
    
    return soft_min / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    # Generate initial configurations
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Two rings
    for r_in in [0.3, 0.5, 0.7]:
        for phi in [0.0, np.pi/4, np.pi/7]:
            angles_out = np.linspace(0, 2*np.pi, 10, endpoint=False)
            angles_in = np.linspace(0, 2*np.pi, 4, endpoint=False) + phi
            pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
            pts_in = np.column_stack((r_in*np.cos(angles_in), r_in*np.sin(angles_in)))
            inits.append(np.vstack((pts_out, pts_in)))
            
    # 3. Random points
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
        
    # Optimization loop
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Phase 1: L-BFGS-B on soft score
        def neg_soft(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp=0.05)
            
        try:
            res1 = minimize(neg_soft, init.flatten(), method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            p1 = res1.x.reshape(N, 2)
            s1 = _calculate_score(p1)
            try_update(s1, p1)
        except Exception:
            continue
            
        # Phase 2: Nelder-Mead on hard score
        def neg_hard(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
            
        try:
            res2 = minimize(neg_hard, p1.flatten(), method='Nelder-Mead',
                            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            p2 = res2.x.reshape(N, 2)
            s2 = _calculate_score(p2)
            try_update(s2, p2)
        except Exception:
            pass

    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 2:
        def neg_hard(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        try:
            res3 = minimize(neg_hard, best_points.flatten(), method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            p3 = res3.x.reshape(N, 2)
            s3 = _calculate_score(p3)
            try_update(s3, p3)
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
