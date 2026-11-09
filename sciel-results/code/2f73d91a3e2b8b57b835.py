import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def generate_hexagonal_guesses():
    """
    Generate a few heuristic configurations based on hexagonal packing.
    """
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
    
    # Guess 2: Hexagonal-like (1 center, 6 ring, 4 corners adjusted)
    # This is tricky to get exactly right with fixed counts, so we use a grid
    # and pick 11 points.
    
    # Create a fine hexagonal grid
    # Side length of big triangle is 1.
    # Small triangle side s.
    # Number of points ~ 1/s^2.
    # For 11 points, s ~ 1/sqrt(11) ~ 0.3.
    # Let's try s = 0.25 (16 points approx) and pick 11 best?
    # Or just define a 4-row grid: 4, 3, 2, 1 = 10 points + 1 center?
    
    # Let's try a 4-3-2-1 layout with optimized y positions
    # y for row i (0-indexed from bottom)
    # Row 0 (4 pts), Row 1 (3 pts), Row 2 (2 pts), Row 3 (1 pt)
    # Total 10. Need 11.
    # Maybe 5-4-2? Or 4-4-3?
    
    # Guess 3: 4-4-3 layout
    y_levels_2 = [0.1*h, 0.4*h, 0.7*h]
    counts_2 = [4, 4, 3]
    pts = []
    for y, c in zip(y_levels_2, counts_2):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.02 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
    
    # Guess 4: 5-3-2-1 but shifted
    y_levels_3 = [0.05*h, 0.3*h, 0.6*h, 0.9*h]
    counts_3 = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels_3, counts_3):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.05 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
            
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    # 1. Local optimization on heuristics
    guesses = generate_hexagonal_guesses()
    
    # Add random guesses
    rng = np.random.default_rng(42)
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    
    for start_points in guesses:
        if time.time() > deadline - 30: # Leave time for DE
            break
        start_points = project_to_triangle(start_points)
        x0 = start_points.flatten()
        
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 
                                   'adaptive': True, 'initial_simplex': 0.01})
            
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # 2. Differential Evolution
    if time.time() < deadline - 10.0:
        bounds = [(0.0, 1.0), (0.0, h)] * N
        try:
            # DE is stochastic. Use a seed.
            # maxiter is a soft limit, we rely on time check outside or inside?
            # DE doesn't have a time limit option. We must wrap it or just let it run
            # and check time after. But if it hangs, we fail.
            # DE iterations are usually fast enough for 22 dims.
            # Let's set a reasonable maxiter.
            res_de = differential_evolution(
                objective_flat, 
                bounds, 
                seed=123, 
                maxiter=200, # Limit to avoid hanging
                tol=1e-6,
                polish=False
            )
            
            curr_points = project_to_triangle(res_de.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # 3. Final Polish with NM
    if best_points is not None and time.time() < deadline - 2.0:
        x0 = best_points.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = project_to_triangle(guesses[0])
        
    return best_points

# EVOLVE-BLOCK-END
