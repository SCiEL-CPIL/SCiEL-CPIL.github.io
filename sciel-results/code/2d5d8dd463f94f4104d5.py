"""
Heilbronn triangles: maximize the smallest triangle area for 11 points.
Strategy: Derivative-Free Active-Set Local Search (Direct Search).

The current program relies on symmetry assumptions and standard optimizers (DE, Nelder-Mead)
on the full or reduced parameter space. This approach is different:
1. It ignores symmetry assumptions entirely, treating all 11 points as free variables 
   (constrained to the triangle via barycentric coordinates or projection).
2. It uses a specific local search heuristic: "Active Set" optimization.
   - At each step, identify the triplet of points (i, j, k) that forms the minimum area triangle.
   - Attempt to increase the area of *this specific* triangle by moving one of its vertices
     in the direction that maximizes the area increase, while ensuring the point stays inside
     the triangle and doesn't drastically reduce other triangle areas (handled by checking 
     the global min after the move).
3. This is a form of coordinate ascent on the non-smooth objective, which often outperforms 
   general-purpose optimizers on packing problems because it directly targets the bottleneck.

We combine this with:
- Multiple random restarts.
- A "shake" mechanism (random perturbations) to escape local minima.
- Efficient vectorized evaluation of the minimum area.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for fast min-area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area_and_idx(pts):
    """
    Compute the minimum area of any triangle formed by three points, 
    and return the index of the triplet that achieves it.
    """
    p_i = pts[_i_idx]
    p_j = pts[_i_idx] # Correction: p_j should be pts[_j_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    return areas[min_idx], min_idx

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    if np.any(y < -eps): return False
    if np.any(y > np.sqrt(3) * x + eps): return False
    if np.any(y > np.sqrt(3) * (1 - x) + eps): return False
    if np.any(x < -eps): return False
    if np.any(x > 1 + eps): return False
    
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _greedy_init(rng):
    """
    Initialize points using a greedy approach to spread them out.
    Start with vertices, then add points that maximize the distance to existing points.
    """
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, S3H]
    
    for i in range(3, N):
        best_pt = None
        best_dist = -1
        # Sample many random points to find the one farthest from existing
        for _ in range(200):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            cand = np.array([x, y])
            
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts[:i] - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        pts[i] = best_pt
    
    return _project_to_triangle(pts)

def _local_search(pts, rng, time_limit):
    """
    Active-set local search.
    Iteratively tries to increase the minimum area by moving the vertices of the 
    minimum-area triangle.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_min_area, min_idx = _min_triangle_area_and_idx(current_pts)
    
    # Get the indices of the worst triangle
    i, j, k = _triplets[min_idx]
    
    # Directions to try for each point in the worst triangle
    # We try moving each point in several directions
    # Directions: 
    # 1. Away from the other two points (increases area)
    # 2. Random perturbations
    # 3. Towards the centroid of the triangle formed by the other two and the triangle center?
    
    # Pre-compute some directions
    directions = []
    # Random directions
    for _ in range(10):
        ang = rng.uniform(0, 2 * np.pi)
        directions.append(np.array([np.cos(ang), np.sin(ang)]))
    
    # Specific directions based on geometry
    # For point i, the area is 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # To increase area, move i in direction perpendicular to (p_j - p_i) or (p_k - p_i)
    # Or simply move i away from the line segment jk.
    
    max_iter = 5000
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        
        if no_improve_count > max_no_improve:
            break
            
        # Identify current worst triangle
        current_min_area, min_idx = _min_triangle_area_and_idx(current_pts)
        i, j, k = _triplets[min_idx]
        
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        # Try to improve by moving i, j, or k
        improved = False
        
        # Direction 1: Move i away from segment jk
        # Vector jk
        v_jk = p_k - p_j
        # Normal to jk
        n_jk = np.array([-v_jk[1], v_jk[0]])
        # Normalize
        norm_n = np.linalg.norm(n_jk)
        if norm_n > 1e-9:
            n_jk /= norm_n
        # Determine which side of jk i is on
        # Cross product (jk x ji)
        v_ji = p_i - p_j
        cross_val = v_jk[0] * v_ji[1] - v_jk[1] * v_ji[0]
        if cross_val < 0:
            n_jk = -n_jk
            
        # Candidate move for i
        new_i = p_i + step_size * n_jk
        new_pts = current_pts.copy()
        new_pts[i] = new_i
        new_pts = _project_to_triangle(new_pts)
        
        new_min_area, _ = _min_triangle_area_and_idx(new_pts)
        if new_min_area > current_min_area + 1e-12:
            current_pts = new_pts
            current_min_area = new_min_area
            improved = True
            no_improve_count = 0
            continue
        else:
            # Try other directions for i
            for d in directions:
                new_i = p_i + step_size * d
                new_pts = current_pts.copy()
                new_pts[i] = new_i
                new_pts = _project_to_triangle(new_pts)
                
                new_min_area, _ = _min_triangle_area_and_idx(new_pts)
                if new_min_area > current_min_area + 1e-12:
                    current_pts = new_pts
                    current_min_area = new_min_area
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                continue
        
        # If i didn't improve, try j
        # Similar logic for j and k
        # For simplicity and speed, we'll just try random perturbations for j and k if i fails
        for p_idx in [j, k]:
            for d in directions:
                new_p = current_pts[p_idx] + step_size * d
                new_pts = current_pts.copy()
                new_pts[p_idx] = new_p
                new_pts = _project_to_triangle(new_pts)
                
                new_min_area, _ = _min_triangle_area_and_idx(new_pts)
                if new_min_area > current_min_area + 1e-12:
                    current_pts = new_pts
                    current_min_area = new_min_area
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
                
        if not improved:
            no_improve_count += 1
            # Shake: random small perturbation to all points to escape local min
            if no_improve_count % 10 == 0:
                current_pts = current_pts + rng.uniform(-0.01, 0.01, size=current_pts.shape)
                current_pts = _project_to_triangle(current_pts)
                current_min_area, _ = _min_triangle_area_and_idx(current_pts)
            # Reduce step size occasionally
            if no_improve_count % 20 == 0:
                step_size *= 0.9
                
    return current_pts, current_min_area

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    def _update_best(pts):
        nonlocal best_pts, best_score
        pts = _project_to_triangle(pts)
        if _check_inside(pts):
            score, _ = _min_triangle_area_and_idx(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                return True
        return False

    # Initial structures
    # 1. Greedy init
    init1 = _greedy_init(rng)
    _update_best(init1)
    
    # 2. Symmetric init (from previous strategy, as a good starting point)
    f = 0.5
    r = 0.2
    theta = np.pi / 6
    d1 = 0.1
    d2 = 0.1
    pts_sym = np.zeros((N, 2))
    pts_sym[0] = [0.0, 0.0]
    pts_sym[1] = [1.0, 0.0]
    pts_sym[2] = [0.5, S3H]
    pts_sym[3] = [f, 0.0]
    pts_sym[4] = [1.0 - 0.5*f, S3H*f]
    pts_sym[5] = [0.5*(1.0-f), S3H*(1.0-f)]
    cx, cy = 0.5, S3H/3
    angles = [theta, theta + 2*np.pi/3, theta + 4*np.pi/3]
    for i, ang in enumerate(angles):
        pts_sym[6+i] = [cx + r*np.cos(ang), cy + r*np.sin(ang)]
    pts_sym[9] = [0.5, cy + d1]
    pts_sym[10] = [0.5, cy - d2]
    _update_best(pts_sym)
    
    # 3. Random inits
    inits = [init1, pts_sym]
    for _ in range(5):
        pts_rand = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            pts_rand[i] = [x, y]
        inits.append(pts_rand)
    
    # Run local search on each init
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            res_pts, res_score = _local_search(init_pts, rng, DEADLINE - 5)
            _update_best(res_pts)
        except Exception:
            pass
        
    # If time remains, do a few more random restarts with local search
    while time.time() - start_time < DEADLINE - 10:
        pts_rand = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            pts_rand[i] = [x, y]
            
        try:
            res_pts, res_score = _local_search(pts_rand, rng, DEADLINE - 5)
            _update_best(res_pts)
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    score, _ = _min_triangle_area_and_idx(pts)
    print(f"Score: {score}")

