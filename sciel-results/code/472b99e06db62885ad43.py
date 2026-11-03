"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _barycentric_to_cartesian(u, v):
    """Convert barycentric coords (u,v) to Cartesian.
    u = weight of V1, v = weight of V2, weight of V0 = 1-u-v.
    p = u*V1 + v*V2 (since V0=0)
    """
    return u * V1 + v * V2

def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coords (u, v).
    p = u*V1 + v*V2
    px = u + 0.5*v
    py = (sqrt(3)/2)*v
    v = 2*py/sqrt(3)
    u = px - 0.5*v
    """
    px, py = p
    v = 2.0 * py / np.sqrt(3)
    u = px - 0.5 * v
    return u, v

def _clamp_barycentric(u, v):
    """Clamp barycentric coords so that u>=0, v>=0, u+v<=1."""
    # Use a simple iterative projection
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        scale = 1.0 / (u + v)
        u *= scale
        v *= scale
    return u, v

def _neg_min_area_barycentric(x):
    """Objective function in barycentric coordinates.
    x is a flat array of length 2*N representing (u1,v1,...,uN,vN).
    """
    n = len(x) // 2
    penalty = 0.0
    u = x[0::2]
    v = x[1::2]
    
    # Clamp to valid barycentric region
    for i in range(n):
        if u[i] < 0 or v[i] < 0 or u[i] + v[i] > 1:
            penalty += 1000.0
    
    # Convert to Cartesian
    pts = np.zeros((n, 2))
    for i in range(n):
        pts[i] = _barycentric_to_cartesian(u[i], v[i])
    
    area = min_triangle_area(pts)
    return -area + penalty

def _project_point_to_triangle(p):
    """Project a point onto the triangle using barycentric coords."""
    if point_in_triangle(p):
        return p
    u, v = _cartesian_to_barycentric(p)
    u, v = _clamp_barycentric(u, v)
    return _barycentric_to_cartesian(u, v)

def _generate_lattice_init(seed):
    """Generate initial points on a triangular lattice with jitter."""
    rng = np.random.RandomState(seed)
    pts = []
    
    # Use a structured grid in barycentric coordinates
    # We want approximately 11 points
    # Try different grid structures
    
    # Structure 1: 4x3 grid in barycentric (u,v) space
    # u from 0 to 1, v from 0 to 1-u
    candidates = []
    for i in range(5):
        for j in range(5):
            u = i / 4.0
            v = j / 4.0 * (1.0 - u)
            if u >= -1e-12 and v >= -1e-12 and u + v <= 1.0 + 1e-12:
                p = _barycentric_to_cartesian(u, v)
                if point_in_triangle(p, eps=1e-8):
                    candidates.append(p)
    
    # Select 11 most evenly spread points
    if len(candidates) >= 11:
        # Use a simple greedy selection
        selected = []
        remaining = list(range(len(candidates)))
        
        # Start with a corner
        start_idx = 0
        selected.append(candidates[start_idx])
        remaining.remove(start_idx)
        
        while len(selected) < 11 and remaining:
            best_idx = 0
            best_dist = 0.0
            for idx in remaining:
                min_d = float('inf')
                for s in selected:
                    d = np.linalg.norm(candidates[idx] - s)
                    min_d = min(min_d, d)
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = idx
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)
        
        pts = selected
    
    if len(pts) != 11:
        # Fallback to random
        pts = []
        for _ in range(11):
            u = rng.random()
            v = rng.random() * (1.0 - u)
            p = _barycentric_to_cartesian(u, v)
            pts.append(p)
    
    # Add small jitter
    jittered = np.array(pts) + rng.normal(0, 0.02, (11, 2))
    
    # Project back to triangle
    for i in range(11):
        jittered[i] = _project_point_to_triangle(jittered[i])
    
    return jittered

def _generate_random_init(seed):
    """Generate random initial points in the triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random() * (1.0 - u)
        p = _barycentric_to_cartesian(u, v)
        pts.append(p)
    return np.array(pts)

def _generate_median_init(seed):
    """Generate points along medians and center."""
    rng = np.random.RandomState(seed)
    v0, v1, v2 = V0, V1, V2
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    c = (v0 + v1 + v2) / 3
    
    pts = [v0, v1, v2, m01, m12, m20, c]
    
    # Add points along medians
    for t in [0.25, 0.75]:
        pts.append(v0 + t * (m12 - v0))
        pts.append(v1 + t * (m20 - v1))
    
    # Take first 11
    pts = pts[:11]
    pts = np.array(pts)
    
    # Add small jitter
    pts = pts + rng.normal(0, 0.03, (11, 2))
    
    # Project back
    for i in range(11):
        pts[i] = _project_point_to_triangle(pts[i])
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # Lattice-based inits with different seeds
    for seed in [42, 123, 456, 789]:
        inits.append(_generate_lattice_init(seed))
    
    # Random inits
    for seed in [1, 2, 3, 4]:
        inits.append(_generate_random_init(seed))
    
    # Median-based inits
    for seed in [10, 20]:
        inits.append(_generate_median_init(seed))
    
    # Convert initializations to barycentric parameter space
    bary_inits = []
    for init_pts in inits:
        params = np.zeros(2 * N)
        for i in range(N):
            u, v = _cartesian_to_barycentric(init_pts[i])
            u, v = _clamp_barycentric(u, v)
            params[2 * i] = u
            params[2 * i + 1] = v
        bary_inits.append(params)
    
    # Run Nelder-Mead optimization on each initialization
    for i, x0 in enumerate(bary_inits):
        if time.time() > deadline - 10.0:
            break
        
        try:
            result = minimize(
                _neg_min_area_barycentric,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True,
                    'initial_simplex': None
                }
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        # Extract and validate result
        x_opt = result.x
        n = len(x_opt) // 2
        pts = np.zeros((n, 2))
        valid = True
        for j in range(n):
            u = x_opt[2 * j]
            v = x_opt[2 * j + 1]
            u, v = _clamp_barycentric(u, v)
            p = _barycentric_to_cartesian(u, v)
            if not point_in_triangle(p):
                p = _project_point_to_triangle(p)
                if not point_in_triangle(p):
                    valid = False
                    break
            pts[j] = p
        
        if valid:
            score = min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback to the first initialization
        best_points = inits[0]
    
    return best_points
