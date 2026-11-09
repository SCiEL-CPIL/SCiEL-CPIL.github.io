import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
h = np.sqrt(3) / 2

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

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
    for _ in range(5):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x  =>  sqrt(3)x - y >= 0
        # Violation: y > sqrt(3)x.
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            # Projection of (px, py) onto line through origin with direction (1, sqrt(3))
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
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

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
    # 1. Use differential evolution for global exploration (structural change).
    # 2. Use Nelder-Mead for local refinement (from current program).
    # 3. Use multiple random restarts if time permits.
    
    # Heuristic Start:
    # Let's define a good symmetric initial guess.
    # 11 points.
    # Symmetric arrangement.
    # Row 1 (bottom): 5 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Row 4 (top): 1 point
    # Total: 11.
    
    guesses = []
    
    # Guess 1: Grid-like
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

    # Phase 1: Local refinement of heuristic starts
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

    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 20.0:
        bounds = [(0.0, 1.0), (0.0, h)] * N
        try:
            de_res = differential_evolution(objective_flat, bounds, 
                                            maxiter=100, 
                                            popsize=15, 
                                            tol=0.01, 
                                            seed=42,
                                            polish=False)
            curr_area = -de_res.fun
            if curr_area > best_score:
                best_score = curr_area
                best_points = project_to_triangle(de_res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 3: Random restarts with Nelder-Mead
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
