import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 148.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(pts):
    """Compute all triangle areas."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_area(pts, beta=100.0):
    """Smooth approximation of minimum area using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    # Clip to avoid log(0)
    areas = np.maximum(areas, 1e-15)
    # Soft min = -log-sum-exp(-beta * areas) / beta
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: shape (N, 3) where each row sums to 1 and all elements >= 0.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    P = b1*A + b2*B + b3*C, A=(0,0), B=(1,0), C=(0.5, S3H)
    x = b2 + 0.5*b3, y = S3H*b3
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates relative to A(0,0), B(1,0), C(0.5, S3H)."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _bary_penalty(bary):
    """Penalty for barycentric coordinates that are outside [0,1] or don't sum to 1."""
    # Penalty for negative coords
    neg_penalty = np.sum(np.minimum(bary, 0)**2)
    # Penalty for sum not equal to 1
    sum_penalty = np.sum((np.sum(bary, axis=1) - 1.0)**2)
    return neg_penalty + sum_penalty

def _objective_bary(x_flat, beta=100.0):
    """
    Objective function in barycentric coordinates.
    x_flat: shape (N*3,) - barycentric coords for N points.
    Returns negative soft-min area + penalty (to minimize).
    """
    bary = x_flat.reshape(N, 3)
    pts = _barycentric_to_cartesian(bary)
    soft_min = _soft_min_area(pts, beta)
    pen = _bary_penalty(bary)
    return -soft_min + 10.0 * pen

def _generate_lattice_initializations():
    """Generate diverse lattice-based initializations for multi-start."""
    inits = []
    
    # 1. Structured: vertices, midpoints, center
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    inits.append(np.array(p_struct))
    
    # 2. Regular grid in barycentric space
    # Use a 3x3 grid in barycentric coords, pick 11 points
    bary_grid = []
    for i in range(4):
        for j in range(4):
            k = 4 - i - j
            if k >= 0:
                bary_grid.append([i/3.0, j/3.0, k/3.0])
    # Pick 11 well-spread points from the grid
    rng = random.Random(123)
    rng.shuffle(bary_grid)
    if len(bary_grid) >= 11:
        selected_bary = np.array(bary_grid[:11])
        pts_grid = _barycentric_to_cartesian(selected_bary)
        inits.append(_project_to_triangle(pts_grid))
    
    # 3. Random with seed
    rng2 = random.Random(456)
    p_rand = np.array([_random_point_in_triangle(rng2) for _ in range(N)])
    inits.append(_project_to_triangle(p_rand))
    
    # 4. Another structured pattern
    p_struct2 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.33, 0.0], [0.67, 0.0],
        [0.17, S3H/3], [0.83, S3H/3],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
        [0.25, S3H/6], [0.75, S3H/6]
    ]
    inits.append(_project_to_triangle(np.array(p_struct2)))
    
    # 5. Perturbed structured
    rng3 = random.Random(789)
    p_pert = np.array(p_struct)
    for i in range(N):
        p_pert[i] = p_pert[i] + rng3.uniform(-0.05, 0.05)
    inits.append(_project_to_triangle(p_pert))
    
    return inits

def _nelder_mead_optimize(init_pts, beta=100.0, max_iter=200):
    """
    Run Nelder-Mead on barycentric coordinates to maximize soft-min area.
    Returns optimized Cartesian points.
    """
    bary = _cartesian_to_bary(init_pts)
    # Ensure valid barycentric coords
    bary = np.maximum(bary, 0)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    x0 = bary.flatten()
    
    result = minimize(
        _objective_bary,
        x0,
        args=(beta,),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    if result.success or result.fun < 0:
        bary_opt = result.x.reshape(N, 3)
        # Project to valid barycentric
        bary_opt = np.maximum(bary_opt, 0)
        bary_opt = bary_opt / np.sum(bary_opt, axis=1, keepdims=True)
        pts_opt = _barycentric_to_cartesian(bary_opt)
        pts_opt = _project_to_triangle(pts_opt)
        return pts_opt
    
    return init_pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses multi-start Nelder-Mead with soft-minimum objective in barycentric coordinates.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Generate diverse initializations
    initializations = _generate_lattice_initializations()
    
    # Evaluate initializations
    for init_pts in initializations:
        score = _min_triangle_area(init_pts)
        if score > best_score:
            best_score = score
            best_pts = init_pts.copy()
    
    # Multi-start Nelder-Mead optimization
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        try:
            opt_pts = _nelder_mead_optimize(init_pts, beta=100.0, max_iter=150)
            if _check_inside(opt_pts):
                score = _min_triangle_area(opt_pts)
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # If we still have time, try more random restarts
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        max_restarts = min(int(remaining_time / 3), 10)
        for r in range(max_restarts):
            if time.time() - start_time > DEADLINE - 3:
                break
            p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            p_rand = _project_to_triangle(p_rand)
            try:
                opt_pts = _nelder_mead_optimize(p_rand, beta=100.0, max_iter=100)
                if _check_inside(opt_pts):
                    score = _min_triangle_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final check
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
    
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
