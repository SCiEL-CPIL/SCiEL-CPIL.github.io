"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def project_to_triangle(p):
    x, y = p
    if y < 0:
        return np.array([x, 0.0])
    if np.sqrt(3) * x - y < 0:
        t = np.dot(p, V2) / np.dot(V2, V2)
        if t < 0: t = 0
        if t > 1: t = 1
        return t * V2
    if -np.sqrt(3) * x - y + np.sqrt(3) < 0:
        v = V2 - V1
        t = np.dot(p - V1, v) / np.dot(v, v)
        if t < 0: t = 0
        if t > 1: t = 1
        return V1 + t * v
    return p

def min_triangle_area(points):
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                a = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if a < min_a:
                    min_a = a
    return min_a

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    pts = []
    for i in range(4):
        for j in range(3-i):
            p = (i/3.0) * V1 + (j/3.0) * V2 + (1 - (i+j)/3.0) * V0
            pts.append(p)
    
    p10 = (pts[4] + pts[1] + pts[5]) / 3.0
    pts.append(p10)
    
    points = np.array(pts)
    best_points = points.copy()
    best_area = min_triangle_area(points)
    
    def objective(flat_pts):
        p = flat_pts.reshape(-1, 2)
        for i in range(len(p)):
            p[i] = project_to_triangle(p[i])
        return -min_triangle_area(p)
        
    x0 = points.flatten()
    
    # Run optimization
    while time.time() < deadline:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        except Exception:
            break
            
        new_points = res.x.reshape(-1, 2)
        for i in range(len(new_points)):
            new_points[i] = project_to_triangle(new_points[i])
            
        new_area = min_triangle_area(new_points)
        
        if new_area > best_area:
            best_area = new_area
            best_points = new_points
            x0 = new_points.flatten()
        else:
            if time.time() > deadline - 5:
                break
            # Perturb to escape local minima
            np.random.seed(int(time.time() * 1000) % 10000)
            pert = np.random.normal(0, 0.05, size=x0.shape)
            x0 = x0 + pert
            p_flat = x0.reshape(-1, 2)
            for i in range(len(p_flat)):
                p_flat[i] = project_to_triangle(p_flat[i])
            x0 = p_flat.flatten()

    return best_points
# EVOLVE-BLOCK-END
