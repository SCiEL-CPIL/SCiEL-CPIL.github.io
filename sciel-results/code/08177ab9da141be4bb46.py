import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha < 1e-10:
        return -1e9
    ma = _min_triangle_area(points)
    return ma / ha

def _objective(x):
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    pts[1] = [1, 0]
    for i in range(2, N):
        pts[i] = [x[2*(i-2)], x[2*(i-2)+1]]
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_points = None
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    v = reg_pts[1] - reg_pts[0]
    norm_v = np.linalg.norm(v)
    theta = np.arctan2(v[1], v[0])
    rot = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    scaled_rot_pts = (reg_pts - reg_pts[0]) * rot / norm_v
    x0 = scaled_rot_pts[2:].flatten()
    
    # Run optimization
    if time.time() < deadline:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < -best_score:
                best_score = -res.fun
                pts = np.zeros((N, 2))
                pts[0] = [0, 0]
                pts[1] = [1, 0]
                for i in range(2, N):
                    pts[i] = [res.x[2*(i-2)], res.x[2*(i-2)+1]]
                best_points = pts
        except Exception:
            pass

    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() >= deadline:
            break
        
        r = np.sqrt(rng.random(N))
        theta_rand = rng.random(N) * 2 * np.pi
        rand_pts = np.column_stack([r * np.cos(theta_rand), r * np.sin(theta_rand)])
        
        if np.linalg.norm(rand_pts[0] - rand_pts[1]) < 1e-6:
            continue
            
        v = rand_pts[1] - rand_pts[0]
        norm_v = np.linalg.norm(v)
        theta_rot = np.arctan2(v[1], v[0])
        rot = np.array([[np.cos(-theta_rot), -np.sin(-theta_rot)], [np.sin(-theta_rot), np.cos(-theta_rot)]])
        scaled_rot_pts = (rand_pts - rand_pts[0]) * rot / norm_v
        x0 = scaled_rot_pts[2:].flatten()
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < -best_score:
                best_score = -res.fun
                pts = np.zeros((N, 2))
                pts[0] = [0, 0]
                pts[1] = [1, 0]
                for i in range(2, N):
                    pts[i] = [res.x[2*(i-2)], res.x[2*(i-2)+1]]
                best_points = pts
        except Exception:
            pass

    if best_points is None:
        return reg_pts

    return best_points
