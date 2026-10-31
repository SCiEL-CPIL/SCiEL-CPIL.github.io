import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def normalize(points):
    # points shape (N, 2)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    try:
        hull = ConvexHull(points_centered)
        area = hull.volume
        if area < 1e-10:
             return points_centered
        points_scaled = points_centered / np.sqrt(area)
    except Exception:
        # Fallback if hull fails (e.g., collinear)
        # Use RMS radius as proxy
        r = np.sqrt(np.mean(np.sum(points_centered**2, axis=1)))
        if r < 1e-10:
             return points_centered
        points_scaled = points_centered / r
    return points_scaled

def min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def score(points):
    pts = normalize(points)
    return min_triangle_area(pts)

def objective(x):
    pts = x.reshape(N, 2)
    return -score(pts)

def construct():
    start_time = time.time()
    best_score = -1
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init1)
    
    # 2. Center + 2 rings
    init2 = np.zeros((N, 2))
    # 1 center
    # 6 inner
    for i in range(6):
        a = 2*np.pi*i/6
        init2[1+i] = [0.5*np.cos(a), 0.5*np.sin(a)]
    # 6 outer
    for i in range(6):
        a = 2*np.pi*i/6
        init2[7+i] = [1.0*np.cos(a), 1.0*np.sin(a)]
    inits.append(init2)
    
    # 3. Random
    rng = np.random.RandomState(42)
    for _ in range(10):
        inits.append(rng.random((N, 2)))
        
    # Optimize each init
    for init in inits:
        if time.time() - start_time > 100: break
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < -best_score:
                # Normalize the result before scoring/returning
                pts = normalize(res.x.reshape(N, 2))
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception as e:
            pass
            
    # Polish best point with Powell or tighter NM
    if best_points is not None:
        if time.time() - start_time < 110:
             try:
                res = minimize(objective, best_points.flatten(), method='Powell', 
                               options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12})
                pts = normalize(res.x.reshape(N, 2))
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
             except:
                pass
                
    if best_points is None:
        # Fallback to regular polygon
        best_points = normalize(inits[0])
        
    return best_points
