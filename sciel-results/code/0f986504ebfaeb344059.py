import numpy as np
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

def project_to_triangle(p):
    x, y = p[0], p[1]
    
    if y < 0:
        y = 0
        
    if y > np.sqrt(3) * x:
        x = y / np.sqrt(3)
        
    if y > -np.sqrt(3) * (x - 1):
        x = 1 - y / np.sqrt(3)
        
    if x < 0:
        x = 0
    if x > 1:
        x = 1
        
    return np.array([x, y])

def min_triangle_area(points):
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                 p2[0] * (p3[1] - p1[1]) + 
                                 p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    points = params.reshape(N, 2)
    projected_points = np.array([project_to_triangle(p) for p in points])
    area = min_triangle_area(projected_points)
    return -area

def construct():
    h = H
    # Base grid points
    base_points = np.array([
        [0.0, 0.0],
        [1.0/3, 0.0],
        [2.0/3, 0.0],
        [1.0, 0.0],
        [1.0/6, h/3],
        [0.5, h/3],
        [5.0/6, h/3],
        [1.0/3, 2.0*h/3],
        [2.0/3, 2.0*h/3],
        [0.5, h],
        [7.0/18, h/9]
    ])
    
    best_points = None
    best_score = -float('inf')
    
    # Run 1: From base grid
    x0 = base_points.flatten()
    res1 = minimize(objective, x0, method='Nelder-Mead', 
                    options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 10000})
    pts1 = np.array([project_to_triangle(p) for p in res1.x.reshape(N, 2)])
    score1 = min_triangle_area(pts1)
    
    if score1 > best_score:
        best_score = score1
        best_points = pts1
        
    # Run 2: From jittered grid
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.02, 0.02, size=(N, 2))
    # Scale jitter to be small relative to the triangle
    jitter[:, 0] *= 0.1
    jitter[:, 1] *= 0.1
    x0_2 = (base_points + jitter).flatten()
    
    res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                    options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 10000})
    pts2 = np.array([project_to_triangle(p) for p in res2.x.reshape(N, 2)])
    score2 = min_triangle_area(pts2)
    
    if score2 > best_score:
        best_score = score2
        best_points = pts2
        
    return best_points
