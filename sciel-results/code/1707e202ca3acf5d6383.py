import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices
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
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    v3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    return np.sum(v1**2 + v2**2 + v3**2)

def _soft_objective(flat_points, beta=100.0):
    """
    Minimizes a smooth approximation of the negative min area.
    Lower is better.
    """
    points = flat_points.reshape(-1, 2)
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Add penalty for being outside
    pen = _penalty(points)
    if pen > 1e-6:
        return 1000.0 + pen
        
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # We want to maximize min area, so minimize -min area.
    # soft_min(areas) approximates min(areas).
    # To avoid underflow/overflow, use log-sum-exp trick.
    # log_sum_exp(-beta * areas) = max(-beta*areas) + log(sum(exp(-beta*areas - max(-beta*areas))))
    # -beta * areas are negative. Let M = max(-beta * areas) = -beta * min(areas).
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    
    vals = -beta * areas
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    # We want to minimize -soft_min_val
    return -soft_min_val

def _project_point(p):
    x, y = p
    # If y < 0
    if y < 0:
        y = 0
    # If y > sqrt(3) * x
    if y > np.sqrt(3) * x:
        # Project onto y = sqrt(3) x
        # Line: sqrt(3)x - y = 0. Normal (sqrt(3), -1). 
        # Or use vector projection.
        # Direction d = (1, sqrt(3)). Unit u = (0.5, sqrt(3)/2).
        # proj = dot(p, u) * u
        dot = x * 0.5 + y * (np.sqrt(3)/2)
        x = dot * 0.5
        y = dot * (np.sqrt(3)/2)
    # If y > sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x):
        # Line: y = -sqrt(3)(x-1) => sqrt(3)x + y - sqrt(3) = 0
        # Direction d = (1, -sqrt(3)). Unit u = (0.5, -sqrt(3)/2).
        # Point on line: (1, 0).
        # v = p - (1,0) = (x-1, y)
        # proj = (1,0) + dot(v, u) * u
        vx = x - 1
        vy = y
        dot = vx * 0.5 + vy * (-np.sqrt(3)/2)
        x = 1 + dot * 0.5
        y = 0 + dot * (-np.sqrt(3)/2)
        
    # Clamp to [0,1] just in case of floating point errors
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    return np.array([x, y])

def _is_valid(points):
    return _penalty(points) < 1e-6

def _generate_lattice():
    """
    Generate a triangular lattice configuration.
    """
    # For N=11, a lattice doesn't fit perfectly, but we can place them in a grid-like structure
    # and then optimize.
    # Simple approach: place points in rows.
    # Row 0: 1 point
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points? 
    # Total 1+2+3+3+2 = 11.
    # Or just use a standard packing layout if known, or random init is fine if SA is good.
    # Let's try a heuristic grid.
    pts = []
    # Center of mass is (0.5, H/3)
    # Let's just scatter them roughly
    # This is a fallback if SA doesn't start well.
    # Actually, random init for SA is often sufficient if the landscape is not too rugged.
    # But a lattice is a good start.
    
    # Let's try to place them in a hexagonal pattern clipped to the triangle.
    # Spacing s.
    # Area of triangle = sqrt(3)/4.
    # Area per point ~ sqrt(3)/4 / 11.
    # s^2 * sqrt(3)/2 ~ sqrt(3)/44 => s^2 ~ 1/22 => s ~ 0.21.
    
    s = 0.21
    # Generate grid points
    grid_pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * s
            y = j * s * np.sqrt(3)
            if i % 2 != 0:
                y += s * np.sqrt(3) / 2
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    
    # Shift to center of triangle
    cx, cy = 0.5, H/3
    grid_pts += [cx, cy]
    
    # Filter points inside triangle
    mask = _penalty(grid_pts) == 0
    inside_pts = grid_pts[mask]
    
    # If we have more than N, select the ones closest to the center? Or just take first N?
    # Better: select a subset that covers the area well.
    # For simplicity, if we have >= N, we can just take the first N and let SA fix it.
    # If < N, add random points.
    
    if len(inside_pts) >= N:
        # Shuffle and take N? Or sort by distance to center?
        # Sorting by distance to center might cluster them.
        # Let's just take a random sample of N from the inside points.
        idx = np.random.choice(len(inside_pts), N, replace=False)
        return inside_pts[idx].copy()
    else:
        # Fill with random
        pts_list = list(inside_pts)
        while len(pts_list) < N:
            # Random point in triangle
            u = np.random.random()
            v = np.random.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts_list.append([x, y])
        return np.array(pts_list)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer
    
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    # 1. Lattice
    candidates.append(_generate_lattice().flatten())
    # 2. Random
    for _ in range(5):
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
        candidates.append(np.array(pts).flatten())
    
    best_score = -np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if _is_valid(pts):
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    # Evaluate initial candidates
    for cand in candidates:
        pts = cand.reshape(-1, 2)
        update_best(pts)
        
    # Simulated Annealing
    # We will run SA on the best candidate found so far, or restart if stuck?
    # Let's run SA for a fixed number of iterations or time.
    
    # Parameters
    T_start = 0.1
    T_end = 1e-5
    alpha = 0.9999 # Cooling rate
    
    # We need to track the best within SA to restart if needed?
    # For now, run SA on the global best.
    
    current_pts = best_points.copy()
    current_score = best_score
    T = T_start
    
    # Pre-compute for speed? No, _min_area is vectorized.
    
    while time.time() < deadline - 20: # Leave 20s for local refinement
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Perturb
        # Step size proportional to T?
        step = 0.05 * (T / T_start + 0.1) # Heuristic step size
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        new_pts = current_pts.copy()
        new_pts[i] = new_pts[i] + [dx, dy]
        
        # Project to triangle
        new_pts[i] = _project_point(new_pts[i])
        
        # Check validity
        if _penalty(new_pts) > 1e-6:
            # Reject if invalid
            T *= alpha
            continue
            
        new_score = _min_area(new_pts)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            # Better
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        else:
            # Worse
            # Accept with probability exp(delta / T)
            # Note: delta is negative.
            # If T is small, exp is small.
            # If delta is large negative, exp is 0.
            p = np.exp(delta / T)
            if rng.random() < p:
                current_pts = new_pts
                current_score = new_score
        
        T *= alpha
        if T < T_end:
            break
            
    # Local Refinement
    # Use the best_points found
    if best_points is not None:
        # Use Nelder-Mead on the soft objective to polish
        # The soft objective is smooth, so it should converge well.
        # We minimize -soft_min_area.
        
        # Wrap objective for minimizer
        def obj_to_min(x):
            return _soft_objective(x, beta=50.0)
            
        try:
            res = minimize(
                obj_to_min, 
                best_points, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            refined_pts = res.x.reshape(-1, 2)
            if _is_valid(refined_pts):
                refined_score = _min_area(refined_pts)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_pts.copy()
        except Exception:
            pass
            
    # Final check
    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(-1, 2)
        
    return best_points
