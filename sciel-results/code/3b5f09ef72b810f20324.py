import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)  # shape (165, 3)

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
            
        # Edge AC: y <= sqrt(3) x
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
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
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def symmetric_parameterization(x):
    """
    x: array of shape (11,)
    Returns 11 points in the triangle, using reflection symmetry.
    
    We parameterize 6 unique points and their mirror images, plus 1 center point.
    Actually, let's use a simpler symmetry:
    The triangle is symmetric about the vertical line x=0.5.
    We can define points such that if (x, y) is a point, (1-x, y) is also a point.
    For 11 points, we can have:
    - 5 pairs of symmetric points (10 points)
    - 1 point on the axis of symmetry (x=0.5)
    
    So we need 5*2 + 1 = 11 parameters? No, we need 5*(x,y) + 1*(y) = 11 parameters.
    Let x = [x1, y1, x2, y2, ..., x5, y5, y6]
    Points:
    (x1, y1), (1-x1, y1)
    (x2, y2), (1-x2, y2)
    ...
    (x5, y5), (1-x5, y5)
    (0.5, y6)
    
    We must ensure all points are inside the triangle.
    For a point (x, y) to be inside:
    0 <= y <= h
    y/sqrt(3) <= x <= 1 - y/sqrt(3)
    
    So for (x_i, y_i):
    y_i/sqrt(3) <= x_i <= 0.5  (assuming x_i <= 0.5)
    And y_i in [0, h]
    For (0.5, y6): y6 in [0, h]
    """
    x1, y1 = x[0], x[1]
    x2, y2 = x[2], x[3]
    x3, y3 = x[4], x[5]
    x4, y4 = x[6], x[7]
    x5, y5 = x[8], x[9]
    y6 = x[10]
    
    # Enforce bounds to keep points inside
    # y_i in [0, h]
    y1 = np.clip(y1, 0.0, h)
    y2 = np.clip(y2, 0.0, h)
    y3 = np.clip(y3, 0.0, h)
    y4 = np.clip(y4, 0.0, h)
    y5 = np.clip(y5, 0.0, h)
    y6 = np.clip(y6, 0.0, h)
    
    # x_i in [y_i/sqrt(3), 0.5]
    x1 = np.clip(x1, y1/np.sqrt(3), 0.5)
    x2 = np.clip(x2, y2/np.sqrt(3), 0.5)
    x3 = np.clip(x3, y3/np.sqrt(3), 0.5)
    x4 = np.clip(x4, y4/np.sqrt(3), 0.5)
    x5 = np.clip(x5, y5/np.sqrt(3), 0.5)
    
    pts = np.array([
        [x1, y1], [1-x1, y1],
        [x2, y2], [1-x2, y2],
        [x3, y3], [1-x3, y3],
        [x4, y4], [1-x4, y4],
        [x5, y5], [1-x5, y5],
        [0.5, y6]
    ])
    
    return pts

def objective_symmetric(x):
    """
    x: array of shape (11,)
    Returns negative min area (to be minimized).
    """
    points = symmetric_parameterization(x)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Bounds for the 11 parameters
    # x_i in [0, 0.5], y_i in [0, h]
    # But we clip in symmetric_parameterization, so we can use wider bounds for optimization
    # Let's use [0, 1] for x and [0, h] for y as initial bounds
    
    # Generate diverse starting points
    starts = []
    
    # 1. Symmetric rows (5, 3, 2, 1) - converted to symmetric parameterization
    # This is tricky to map exactly, so let's just use random starts and some structured ones
    
    # 2. Random starts in the parameter space
    for _ in range(20):
        x = np.zeros(11)
        for i in range(5):
            x[2*i] = rng.uniform(0.1, 0.4)  # x_i
            x[2*i+1] = rng.uniform(0.1*h, 0.9*h)  # y_i
        x[10] = rng.uniform(0.1*h, 0.9*h)  # y6
        starts.append(x)
        
    # 3. Structured starts: points arranged in rows
    # Row 1: 2 points (symmetric pair)
    # Row 2: 2 points (symmetric pair)
    # Row 3: 2 points (symmetric pair)
    # Row 4: 2 points (symmetric pair)
    # Row 5: 2 points (symmetric pair)
    # Center: 1 point
    
    y_levels = [0.1*h, 0.3*h, 0.5*h, 0.7*h, 0.9*h]
    x_offsets = [0.2, 0.3, 0.25, 0.35, 0.2]
    
    x_struct = np.zeros(11)
    for i, (y, xo) in enumerate(zip(y_levels, x_offsets)):
        x_struct[2*i] = 0.5 - xo  # x_i
        x_struct[2*i+1] = y
    x_struct[10] = 0.5*h  # y6
    starts.append(x_struct)
    
    # Another structured start
    y_levels2 = [0.05*h, 0.25*h, 0.45*h, 0.65*h, 0.85*h]
    x_offsets2 = [0.15, 0.25, 0.2, 0.3, 0.1]
    
    x_struct2 = np.zeros(11)
    for i, (y, xo) in enumerate(zip(y_levels2, x_offsets2)):
        x_struct2[2*i] = 0.5 - xo
        x_struct2[2*i+1] = y
    x_struct2[10] = 0.5*h
    starts.append(x_struct2)
    
    # Run Nelder-Mead from each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(objective_symmetric, x0, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 
                                   'adaptive': True})
            
            curr_points = symmetric_parameterization(res.x)
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
    
    # If we have time, do a final polish
    if best_points is not None and time.time() < deadline - 2.0:
        # Invert the parameterization approximately
        # This is complex, so let's just do a few more random restarts with tighter bounds
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            x0 = np.zeros(11)
            for i in range(5):
                x0[2*i] = rng.uniform(0.15, 0.35)
                x0[2*i+1] = rng.uniform(0.2*h, 0.8*h)
            x0[10] = rng.uniform(0.3*h, 0.7*h)
            
            try:
                res = minimize(objective_symmetric, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 
                                       'adaptive': True})
                curr_points = symmetric_parameterization(res.x)
                curr_area = min_triangle_area(curr_points)
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = curr_points
            except Exception:
                pass

    # Fallback if nothing worked
    if best_points is None:
        # Use the symmetric rows as fallback
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
        best_points = np.array(pts)

    return best_points

# EVOLVE-BLOCK-END
