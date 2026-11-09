import numpy as np
from scipy.optimize import minimize
import time
import itertools

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
    """
    min_area = float('inf')
    n = len(points)
    # Vectorized approach for speed?
    # With N=11, 165 triangles. A loop is fine, but let's try to be efficient.
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v13 = p3 - p1
                cross = v12[0] * v13[1] - v12[1] * v13[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
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
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    # Strategy:
    # 1. Use a good symmetric initial guess.
    # 2. Use Nelder-Mead with tight tolerances.
    # 3. Use multiple random restarts if time permits.
    # 4. Try to exploit symmetry? The optimal solution for 11 points in an equilateral triangle
    #    is likely symmetric. We can constrain the search to symmetric configurations?
    #    No, let's just use a good heuristic start.
    
    # Heuristic Start:
    # 11 points.
    # Symmetric arrangement.
    # Let's try a "hexagonal" like packing adapted to triangle.
    # Or simply distribute points in rows.
    
    # Guess 1: Symmetric rows
    # Row 1 (bottom): 4 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Row 4 (top): 1 point
    # Total: 10. Need 11.
    # Maybe:
    # Row 1: 5 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Row 4: 1 point
    # Total: 11.
    
    # Let's define y-coordinates for rows.
    # h = sqrt(3)/2
    # y0 = 0
    # y1 = h/4
    # y2 = h/2
    # y3 = 3h/4
    # y4 = h
    
    # But points on edges might be bad for min area?
    # Actually, placing points on edges is often optimal for packing.
    
    # Let's try to place points at:
    # 1. Vertices: A, B, C (3 points)
    # 2. Midpoints of edges: (0.5, 0), (0.75, h/2), (0.25, h/2) (3 points)
    # 3. Centroid: (0.5, h/3) (1 point)
    # 4. Remaining 4 points?
    #    Maybe midpoints of the lines from vertices to centroid?
    #    Or just jittered random restarts are safer.
    
    # Let's use a set of diverse initial guesses.
    
    guesses = []
    
    # Guess 1: Grid-like
    # 5 points at y=0.1h
    # 3 points at y=0.4h
    # 2 points at y=0.7h
    # 1 point at y=0.9h
    y_levels = [0.1*h, 0.4*h, 0.7*h, 0.9*h]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        # Width at y is 1 - y/h
        w = 1.0 - y/h
        # x ranges from 0.5 - w/2 to 0.5 + w/2
        # But we need to stay inside triangle.
        # At y, x_min = y/sqrt(3), x_max = 1 - y/sqrt(3)
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            xs = np.linspace(x_min + 0.05*(x_max-x_min), x_max - 0.05*(x_max-x_min), c)
        else:
            xs = np.array([(x_min + x_max)/2])
        for x in xs:
            pts.append([x, y])
    guess1 = np.array(pts)
    guesses.append(guess1)
    
    # Guess 2: Random
    rng = np.random.default_rng(42)
    guess2 = rng.random((N, 2)) * np.array([1.0, h])
    guesses.append(guess2)
    
    # Guess 3: Symmetric hexagonal cluster
    # Center at (0.5, h/3)
    # 6 points around it
    # 4 points at corners?
    # Let's just use random restarts primarily.
    
    for i, start_points in enumerate(guesses):
        if time.time() > deadline:
            break
            
        start_points = project_to_triangle(start_points)
        x0 = start_points.flatten()
        
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            curr_area = -res.fun
            if curr_area > best_score:
                best_score = curr_area
                best_points = project_to_triangle(res.x.reshape(N, 2))
        except Exception:
            pass

    # Random restarts
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(123)
        while time.time() < deadline - 2.0:
            start_points = rng2.random((N, 2)) * np.array([1.0, h])
            start_points = project_to_triangle(start_points)
            x0 = start_points.flatten()
            
            try:
                res = minimize(objective_flat, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
                
                curr_area = -res.fun
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = project_to_triangle(res.x.reshape(N, 2))
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = np.array([[0.5, 0.5*h]] * N) # Dummy, will be projected

    return best_points

# EVOLVE-BLOCK-END
