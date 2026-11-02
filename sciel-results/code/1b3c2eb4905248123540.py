"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Direct Search using Vectorized Nelder-Mead (Derivative-Free).
   - Avoids gradient calculations and explicit constraint handling overhead.
   - Robust to the non-smooth nature of the 'minimum area' objective.
2. High-Fidelity Multi-Start:
   - Uses a Greedy Max-Min Distance initialization to create diverse, 
     high-quality starting configurations.
   - Adds random perturbations to these bases for further diversity.
3. Vectorized Area Calculation:
   - Pre-computes triplet indices.
   - Uses NumPy broadcasting to calculate all 165 triangle areas in one shot.
   - This makes each Nelder-Mead function evaluation extremely fast, 
     allowing for more iterations and restarts within the time limit.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

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
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

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
    
    # Clip y
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return np.array([x, y])

def _greedy_init(rng):
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(p0)
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(500):
            cand = _random_point_in_triangle(rng)
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        if best_pos is not None:
            pts.append(best_pos)
    return np.array(pts)

def _structured_init():
    # Vertices
    v0 = [0.0, 0.0]
    v1 = [1.0, 0.0]
    v2 = [0.5, S3H]
    
    # Midpoints
    m01 = [0.5, 0.0]
    m12 = [0.75, S3H/2]
    m20 = [0.25, S3H/2]
    
    # Centroid
    c = [0.5, S3H/3]
    
    base_pts = np.array([v0, v1, v2, m01, m12, m20, c])
    
    # Add 4 more points spread out
    # Quarter points on edges
    q1 = [0.25, 0.0]
    q2 = [0.75, 0.0]
    q3 = [0.125, S3H/4]
    q4 = [0.875, S3H/4]
    
    return np.vstack([base_pts, [q1, q2, q3, q4]])

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Multi-start Nelder-Mead (Derivative-Free) with Greedy and Structured initializations.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    initializations = []
    
    # 1. Structured initialization
    initializations.append(_structured_init())
    
    # 2. Greedy initializations
    for _ in range(20):
        initializations.append(_greedy_init(rng))
        
    # 3. Random initializations with perturbation of structured
    for _ in range(10):
        base = _structured_init()
        noise = np.random.normal(0, 0.05, base.shape)
        perturbed = base + noise
        perturbed = _project_to_triangle(perturbed)
        initializations.append(perturbed)
        
    # Shuffle
    rng.shuffle(initializations)
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Evaluate initial point
        if _check_inside(init_pts):
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
        try:
            res = minimize(
                lambda x: -_min_triangle_area(x.reshape(N, 2)),
                init_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            pts_opt = res.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            # Fallback to initial point if it was valid
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
            continue
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break

    # Fallback
    if best_pts is None:
        best_pts = _structured_init()
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
