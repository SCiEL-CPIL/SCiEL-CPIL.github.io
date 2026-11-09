import numpy as np
from scipy.optimize import minimize, differential_evolution
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
            
        # Edge AC: y <= sqrt(3) x  =>  sqrt(3)x - y >= 0
        # Violation: y > sqrt(3)x
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

def compute_areas(points):
    """
    points: (N, 2)
    Returns array of areas for all triplets.
    """
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    # Cross product in 2D
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    return np.min(compute_areas(points))

def soft_min_triangle_area(points, beta=100.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    min(a_i) = -lim_{beta->inf} (1/beta) * log(sum(exp(-beta * a_i)))
    We want to maximize min area, so we minimize -min area.
    -min(a_i) = lim_{beta->inf} (1/beta) * log(sum(exp(-beta * a_i)))
    """
    areas = compute_areas(points)
    # Shift for numerical stability
    min_a = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_a))
    log_sum = np.log(np.sum(exp_terms))
    # -min(a_i) approx -min_a - (1/beta) * log_sum
    return -min_a - (1.0 / beta) * log_sum

def objective_flat(x, beta=100.0):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area (to be minimized).
    Uses soft min for smoothness.
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return soft_min_triangle_area(points, beta)

def objective_exact(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative exact min area (to be minimized).
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Differential Evolution with Soft Min ---
    # The soft min objective is smooth, allowing DE to explore the landscape more effectively
    # without being confused by the non-differentiable points of the exact min.
    
    bounds = [(0.0, 1.0), (0.0, h)] * N
    
    popsize = 15  # Larger population for better global search
    maxiter = 80  # More iterations since we have a smoother landscape
    
    if time.time() < deadline - 20.0:
        try:
            # Use a moderate beta for the global phase to keep it smooth but not too flat
            res = differential_evolution(
                lambda x: objective_flat(x, beta=50.0), 
                bounds, 
                seed=42, 
                maxiter=maxiter, 
                popsize=popsize, 
                tol=0.01, 
                mutation=(0.5, 1.0), 
                recombination=0.7,
                polish=False  # We will polish later with exact objective
            )
            
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # --- Phase 2: Local Refinement with Nelder-Mead (Exact Objective) ---
    # Now that we have a good global candidate, we switch to the exact (nonsmooth) objective
    # for local refinement. NM can handle nonsmoothness better than gradient-based methods.
    
    starts = []
    
    # 1. The best point from DE
    if best_points is not None:
        starts.append(best_points)
        
    # 2. Symmetric Rows (5, 3, 2, 1)
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
    starts.append(np.array(pts))
    
    # 3. Hexagonal-like
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
    starts.append(np.array(pts))
    
    # 4. Random starts
    for _ in range(5):
        starts.append(rng.random((N, 2)) * np.array([1.0, h]))
    
    for i, start_points in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        start_points = project_to_triangle(start_points)
        x0 = start_points.flatten()
        
        try:
            # Use exact objective for local refinement
            res = minimize(objective_exact, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 
                                   'adaptive': True})
            
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # --- Phase 3: Final Polish ---
    # If we have time, do a very tight NM on the best point found.
    if best_points is not None and time.time() < deadline - 2.0:
        x0 = best_points.flatten()
        try:
            res = minimize(objective_exact, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
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
