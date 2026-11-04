import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap, positive if separated.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_convex_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_convex_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_min_side_fast(inner_params, outer_angle_deg):
    """
    Fast computation of minimum outer side using vectorized support function.
    inner_params: (N, 3) array of [x, y, angle_deg]
    """
    # Precompute all vertices
    all_vertices = np.zeros((N * 6, 2))
    for i in range(N):
        x, y, a = inner_params[i]
        vs = _hex_vertices(x, y, a)
        all_vertices[i*6:(i+1)*6] = vs
    
    # Outer hexagon normals
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Compute max projection in each normal direction
    max_proj = 0.0
    for n_vec in normals:
        projs = np.dot(all_vertices, n_vec)
        max_proj = max(max_proj, np.max(projs))
    
    return 2.0 * max_proj / np.sqrt(3.0)

def _seed_solution():
    """Return the deterministic flower packing."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    inner_arr = np.array(inner)
    side = _compute_min_side_fast(inner_arr, 30.0)
    return inner_arr, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    
    # Get seed
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    
    best_side = seed_side
    best_inner = seed_inner.copy()
    
    # Strategy: Since the flower packing is highly symmetric, we'll try:
    # 1. Small perturbations of the seed with Nelder-Mead (fast, limited)
    # 2. A direct verification that the seed is valid
    
    # First verify seed validity
    penalty = 0.0
    for i in range(N):
        v1 = _hex_vertices(seed_inner[i, 0], seed_inner[i, 1], seed_inner[i, 2])
        for j in range(i+1, N):
            v2 = _hex_vertices(seed_inner[j, 0], seed_inner[j, 1], seed_inner[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                penalty += (0.01 - sep) ** 2
    
    if penalty < 1e-6:
        # Seed is valid, use it
        final_side = best_side + 1e-9
        return [list(row) for row in best_inner], seed_center, final_side, seed_angle
    
    # If seed has issues, try to fix with optimization
    # Use a simpler parameterization: only positions, fix angles to 30
    def _objective_simple(params):
        if time.time() - start_time > DEADLINE:
            return best_side + 1000
        
        inner_params = np.zeros((N, 3))
        inner_params[0] = [0.0, 0.0, 30.0]
        for i in range(1, N):
            idx = (i-1)*2
            inner_params[i, 0] = params[idx]
            inner_params[i, 1] = params[idx+1]
            inner_params[i, 2] = 30.0
        
        side = _compute_min_side_fast(inner_params, 30.0)
        
        # Penalty for overlap
        pen = 0.0
        for i in range(N):
            v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
            for j in range(i+1, N):
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < 0.01:
                    pen += 100.0 * (0.01 - sep) ** 2
        
        return side + pen
    
    try:
        # Initial params from seed
        init_params = np.zeros(20)
        for i in range(1, N):
            idx = (i-1)*2
            init_params[idx] = seed_inner[i, 0]
            init_params[idx+1] = seed_inner[i, 1]
        
        result = minimize(_objective_simple, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False})
        
        if time.time() - start_time < DEADLINE and result.fun < best_side:
            inner_params = np.zeros((N, 3))
            inner_params[0] = [0.0, 0.0, 30.0]
            for i in range(1, N):
                idx = (i-1)*2
                inner_params[i, 0] = result.x[idx]
                inner_params[i, 1] = result.x[idx+1]
                inner_params[i, 2] = 30.0
            
            # Verify
            pen = 0.0
            for i in range(N):
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                for j in range(i+1, N):
                    v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                    sep = _poly_separation(v1, v2)
                    if sep < 0.01:
                        pen += (0.01 - sep) ** 2
            
            if pen < 1e-6:
                new_side = _compute_min_side_fast(inner_params, 30.0)
                if new_side < best_side:
                    best_side = new_side
                    best_inner = inner_params
    except Exception:
        pass
    
    final_side = best_side + 1e-9
    return [list(row) for row in best_inner], seed_center, final_side, seed_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
