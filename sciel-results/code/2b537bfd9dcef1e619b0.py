import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Pre-compute indices for all triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - SQRT3 * x)
    v3 = np.maximum(0, y - SQRT3 * (1 - x))
    return np.sum(v1**2 + v2**2 + v3**2)

def _project_point(p):
    x, y = p
    # If y < 0
    if y < 0:
        y = 0
    # If y > sqrt(3) * x
    if y > SQRT3 * x:
        dot = x * 0.5 + y * (SQRT3/2)
        x = dot * 0.5
        y = dot * (SQRT3/2)
    # If y > sqrt(3) * (1 - x)
    if y > SQRT3 * (1 - x):
        vx = x - 1
        vy = y
        dot = vx * 0.5 + vy * (-SQRT3/2)
        x = 1 + dot * 0.5
        y = 0 + dot * (-SQRT3/2)
        
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    return np.array([x, y])

def _is_valid(points):
    return _penalty(points) < 1e-6

def _soft_min(areas, beta):
    # Soft min using log-sum-exp trick for stability
    vals = -beta * areas
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    return -log_sum_exp / beta

def _soft_objective(flat_points, beta=50.0):
    points = flat_points.reshape(-1, 2)
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    pen = _penalty(points)
    if pen > 1e-6:
        return 1000.0 + pen
        
    soft_min_val = _soft_min(areas, beta)
    return -soft_min_val

def _generate_hex_init():
    """
    Generate a hexagonal lattice initialization.
    """
    pts = []
    # Approximate spacing for 11 points in area sqrt(3)/4
    # Area per point ~ 0.039. Hex cell area ~ s^2 * sqrt(3)/2.
    # s^2 * 0.866 ~ 0.039 => s^2 ~ 0.045 => s ~ 0.21
    
    s = 0.22
    grid_pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s
            y = j * s * SQRT3
            if i % 2 != 0:
                y += s * SQRT3 / 2
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    
    # Center of triangle
    cx, cy = 0.5, H/3
    grid_pts += [cx, cy]
    
    mask = _penalty(grid_pts) == 0
    inside_pts = grid_pts[mask]
    
    if len(inside_pts) >= N:
        # Select N points. To avoid clustering, we can try to maximize min distance
        # but for init, random selection from lattice is okay if we have many.
        # Let's sort by distance to center and take a spread?
        # Actually, just taking a random subset of the lattice points inside is a decent start.
        # Or better: take the N points that are most "spread out".
        # For simplicity, let's just take the first N or random N.
        # Random N might be better to break symmetry.
        rng = np.random.default_rng(123)
        idx = rng.choice(len(inside_pts), N, replace=False)
        return inside_pts[idx].copy()
    else:
        # Fill with random
        pts_list = list(inside_pts)
        rng = np.random.default_rng(123)
        while len(pts_list) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts_list.append([x, y])
        return np.array(pts_list)

def _generate_random_init(seed):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts.append([x, y])
                break
    return np.array(pts)

def _simulated_annealing(points, start_time, deadline, seed):
    """
    Run SA on the given points. Returns best points and score found.
    """
    rng = np.random.default_rng(seed)
    current_pts = points.copy()
    current_score = _min_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.9998
    
    T = T_start
    step_scale = 0.1
    
    while time.time() < deadline - 10:
        # Pick a random point
        i = rng.integers(0, N)
        
        # Perturb
        # Adaptive step size?
        step = step_scale * (T / T_start + 0.2)
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        new_pts = current_pts.copy()
        new_pts[i] = new_pts[i] + [dx, dy]
        new_pts[i] = _project_point(new_pts[i])
        
        if _penalty(new_pts) > 1e-6:
            T *= alpha
            continue
            
        new_score = _min_area(new_pts)
        
        delta = new_score - current_score
        if delta > 0:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            p = np.exp(delta / T)
            if rng.random() < p:
                current_pts = new_pts
                current_score = new_score
        
        T *= alpha
        if T < T_end:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    # Deadline guard: 120s limit, leave 15s buffer for safety
    deadline = start_time + 105.0
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_hex_init())
    for i in range(10):
        candidates.append(_generate_random_init(i))
    
    best_score = -np.inf
    best_points = None
    
    # Evaluate initial candidates
    for cand in candidates:
        if _is_valid(cand):
            score = _min_area(cand)
            if score > best_score:
                best_score = score
                best_points = cand.copy()
                
    # Run SA on the best candidate
    # Also run SA on a few other good candidates in parallel? No, sequential is fine.
    # Let's run SA on the best candidate found so far.
    if best_points is not None:
        sa_pts, sa_score = _simulated_annealing(best_points, start_time, deadline, seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
            
    # Local Refinement
    # Use Nelder-Mead to polish the best point found
    if best_points is not None:
        # Try multiple beta values for soft min
        for beta in [20, 50, 100]:
            def obj_to_min(x):
                return _soft_objective(x, beta=beta)
                
            try:
                res = minimize(
                    obj_to_min, 
                    best_points, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                refined_pts = res.x.reshape(-1, 2)
                if _is_valid(refined_pts):
                    refined_score = _min_area(refined_pts)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_points = refined_pts.copy()
            except Exception:
                pass
                
        # One more SA pass on the refined points?
        if time.time() < deadline - 5:
            sa_pts, sa_score = _simulated_annealing(best_points, start_time, deadline, seed=99)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts

    # Final check
    if best_points is None:
        # Fallback to first candidate
        best_points = candidates[0]
        
    return best_points
