import numpy as np
import time
from scipy.optimize import minimize

N = 11
TRI_AREA = np.sqrt(3) / 4.0

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_triangle_area(points):
    min_area = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def is_inside_triangle(points, tol=1e-6):
    for p in points:
        x, y = p
        if y < -tol or y > np.sqrt(3) * x + tol or y > -np.sqrt(3) * (x - 1) + tol:
            return False
    return True

def objective(x):
    points = x.reshape(-1, 2)
    if not is_inside_triangle(points):
        return 1e6
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -np.inf
    best_points = None
    
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    # Generate diverse starts
    starts = []
    
    # 1. Random starts
    for _ in range(10):
        pts = []
        while len(pts) < N:
            u, v = np.random.random(2)
            if u + v < 1:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                pts.append([x, y])
        starts.append(np.array(pts).flatten())
        
    # 2. Grid-like start
    # Barycentric grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i + j <= 3:
                # i, j are indices in a 4x4 grid (0 to 3)
                # Barycentric coords: (1 - (i+j)/3)/3 ? No.
                # Let's use normalized coords in [0,1] for the triangle vertices
                # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
                # P = (1-u-v)A + uB + vC
                # u, v in [0,1], u+v <= 1
                # Grid u, v in {0, 0.33, 0.66, 1.0}
                pass
                
    # Simpler grid:
    # Divide triangle into smaller triangles
    # Just use a few specific points
    # Center, midpoints, vertices
    # For N=11, a good start is a hexagonal packing
    # Let's generate a hexagonal packing
    hex_pts = []
    # Rows of points
    # Row 0 (base): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Total 10. Need 11.
    # Let's just use random starts. They work well with Powell.
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        # Use Powell for robustness
        res = minimize(objective, x0, method='Powell', bounds=bounds, 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < 1e5: # Valid
            points = res.x.reshape(-1, 2)
            if is_inside_triangle(points):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
    # If no valid points, use a default
    if best_points is None:
        # Fallback: 11 random points
        pts = []
        while len(pts) < N:
            u, v = np.random.random(2)
            if u + v < 1:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                pts.append([x, y])
        best_points = np.array(pts)
        
    return best_points
