import numpy as np
from scipy.optimize import minimize
import time

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    # Clip to bounding box first
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Iterate a few times to handle corners properly
    for _ in range(3):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x  =>  sqrt(3)x - y >= 0
        # Line: sqrt(3)x - y = 0. Normal pointing inward: (sqrt(3), -1).
        # Actually, for point (x,y), violation is y > sqrt(3)x.
        # Project to line y = sqrt(3)x.
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            # Projection of (px, py) onto line through origin with direction (1, sqrt(3))
            # v = (1, sqrt(3)). v.v = 4.
            # dot = px + py*sqrt(3)
            # proj = dot/4 * v
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            # Clamp to segment AC: x in [0, 0.5]
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x) => y + sqrt(3)x <= sqrt(3)
        # Violation: y > sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            # Line passes through (1,0) and (0.5, h).
            # Equation: y = -sqrt(3)(x-1) = sqrt(3)(1-x)
            # Normal vector pointing inward (towards origin side): (sqrt(3), 1).
            # Let's check: At (0.5, 0), y=0 < sqrt(3)(0.5)=0.866. Inward.
            # Vector from point (1,0) to (0.5, h) is (-0.5, h) = (-0.5, sqrt(3)/2).
            # Normal to this is (sqrt(3)/2, 0.5) or (sqrt(3), 1).
            # Let's project onto the line.
            # Line eq: sqrt(3)x + y - sqrt(3) = 0.
            # Distance d = (sqrt(3)px + py - sqrt(3)) / (3+1) = (sqrt(3)px + py - sqrt(3)) / 4
            # Since violation is y > sqrt(3)(1-x) => sqrt(3)x + y > sqrt(3) => d > 0.
            # Projection: P - d * n, where n=(sqrt(3), 1).
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            # Clamp to segment BC: x in [0.5, 1.0]
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
            
    return p

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    Vectorized inner loop for speed.
    """
    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            ks = np.arange(j + 1, n)
            if len(ks) == 0:
                continue
            v13 = points[ks] - p1
            crosses = v12[0] * v13[:, 1] - v12[1] * v13[:, 0]
            areas = 0.5 * np.abs(crosses)
            min_here = np.min(areas)
            if min_here < min_area:
                min_area = min_here
    return min_area

def objective_flat(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area (to be minimized).
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    guesses = []
    
    # Guess 1: Symmetric rows (5, 3, 2, 1)
    y_levels = [0.08*h, 0.35*h, 0.62*h, 0.88*h]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.03 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
    
    # Guess 2: Hexagonal-like
    pts = []
    y0 = 0.05 * h
    for i in range(5):
        x = 0.1 + 0.2 * i
        pts.append([x, y0])
    y1 = 0.3 * h
    for i in range(3):
        x = 0.25 + 0.25 * i
        pts.append([x, y1])
    y2 = 0.55 * h
    for i in range(2):
        x = 0.35 + 0.3 * i
        pts.append([x, y2])
    pts.append([0.5, 0.8 * h])
    guesses.append(np.array(pts))
    
    # Guess 3 & 4: Random
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    
    for i, start_points in enumerate(guesses):
        if time.time() > deadline:
            break
            
        start_points = project_to_triangle(start_points)
        x0 = start_points.flatten()
        
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 
                                   'adaptive': True, 'initial_simplex': 0.01})
            
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # Random restarts
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(123)
        while time.time() < deadline - 5.0:
            start_points = rng2.random((N, 2)) * np.array([1.0, h])
            start_points = project_to_triangle(start_points)
            x0 = start_points.flatten()
            
            try:
                res = minimize(objective_flat, x0, method='Nelder-Mead', 
                               options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-6, 
                                       'adaptive': True})
                
                curr_points = project_to_triangle(res.x.reshape(N, 2))
                curr_area = min_triangle_area(curr_points)
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = curr_points
            except Exception:
                pass

    # Local refinement
    if best_points is not None and time.time() < deadline - 3.0:
        x0 = best_points.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
    
    if best_points is None:
        pts = []
        y_levels = [0.08*h, 0.35*h, 0.62*h, 0.88*h]
        counts = [5, 3, 2, 1]
        for y, c in zip(y_levels, counts):
            x_min = y / np.sqrt(3)
            x_max = 1.0 - y / np.sqrt(3)
            if c > 1:
                margin = 0.03 * (x_max - x_min)
                xs = np.linspace(x_min + margin, x_max - margin, c)
            else:
                xs = np.array([(x_min + x_max) / 2])
            for x in xs:
                pts.append([x, y])
        best_points = np.array(pts)
    
    return best_points
