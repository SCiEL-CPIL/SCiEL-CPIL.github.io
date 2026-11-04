import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
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
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
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

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _min_enclosing_hex_side(inner_params, outer_angle_deg, smooth_beta=0.0):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon, given a fixed outer angle.
    Uses the support function approach. If smooth_beta > 0, uses a soft-max
    (log-sum-exp) surrogate for smoother gradients during optimization.
    """
    n = len(inner_params)
    apothem_factor = math.sqrt(3) / 2.0
    
    # Edge normals for outer hexagon with rotation outer_angle_deg
    normal_angles = np.deg2rad(outer_angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Collect all positive projections / apothem_factor
    vals = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            projections = normals @ v
            for proj in projections:
                if proj > 0:
                    vals.append(proj / apothem_factor)
    
    if not vals:
        return 0.0
    
    vals = np.array(vals)
    
    if smooth_beta > 0:
        # Soft-max: (1/beta) * log(sum(exp(beta * v)))
        # This is smooth and approximates max as beta -> inf
        beta = smooth_beta
        shifted = beta * (vals - vals.max())
        # Clip to avoid overflow
        shifted = np.clip(shifted, -500, 500)
        return vals.max() + math.log(np.sum(np.exp(shifted))) / beta
    else:
        return float(vals.max())

def _compute_side(inner_params, outer_angle_deg, smooth_beta=0.0):
    """Compute minimum side length for given inner params and outer angle."""
    return _min_enclosing_hex_side(inner_params, outer_angle_deg, smooth_beta)

def _overlap_penalty(inner_params, penalty_weight=50.0):
    """Compute penalty for overlapping inner hexagons."""
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.005:
                penalty += penalty_weight * (0.005 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n, smooth_beta=0.0):
    """Objective: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg, smooth_beta)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """Concentric ring seed solution."""
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
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _lattice_packing():
    """
    Generate a hexagonal lattice packing of 11 unit hexagons.
    The 11 hexagons form a centered hexagonal cluster: 1 center + 6 ring + 4 more.
    Actually the natural 11-hex cluster is: center, 6 neighbors, then 4 from the next ring
    placed at specific positions to minimize the bounding hexagon.
    
    For unit hexagons in a hexagonal lattice, adjacent centers are at distance sqrt(3).
    The lattice vectors are:
    v1 = (sqrt(3), 0)
    v2 = (sqrt(3)/2, 3/2)
    """
    sqrt3 = math.sqrt(3)
    # Hexagonal lattice basis
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Centered hexagonal cluster of 19 positions, pick 11 optimally
    # Positions in lattice coordinates (i,j) -> position = i*v1 + j*v2
    # The 19 positions of a hexagonal cluster of radius 2:
    # (0,0), (±1,0),(0,±1),(1,-1),(-1,1),(1,1),(-1,-1),
    # (2,0),(0,2),(-2,0),(0,-2),(2,-1),(-2,1),(1,-2),(-1,2),(2,1),(-2,-1),(1,2),(-1,-2)
    
    lattice_positions = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (1, 1), (-1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2),
        (2, -1), (-2, 1), (1, -2), (-1, 2),
        (2, 1), (-2, -1), (1, 2), (-1, -2)
    ]
    
    # We need 11. The natural 11-cluster: center + 6 neighbors + 4 from second ring
    # Choose 4 from second ring that are most compact
    # The 6 neighbors are at lattice positions: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    # Second ring positions: (1,1),(-1,-1),(2,0),(0,2),(-2,0),(0,-2),(2,-1),(-2,1),(1,-2),(-1,2),(2,1),(-2,-1),(1,2),(-1,-2)
    
    # For minimal bounding hexagon, we want symmetry. Pick 4 second-ring positions
    # that maintain 3-fold or 6-fold symmetry as much as possible.
    # Try: (1,1), (-1,-1), (2,0), (0,-2) - not symmetric
    # Try: (1,1), (-1,-1), (-2,1), (1,-2) - 3-fold symmetric
    # Let's just use the first 11 in a smart order
    
    # Better: use the 7 inner + 4 outer that are closest to center
    # Distance in lattice coords from center:
    # (1,0): 1, (-1,0): 1, (0,1): 1, (0,-1): 1, (1,-1): 1, (-1,1): 1
    # (1,1): sqrt(3)≈1.73, (-1,-1): 1.73, (2,0): 2, (0,2): 2, etc.
    
    # Pick: center + 6 neighbors + (1,1),(-1,-1),(2,0),(0,-2)
    selected = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,-2)]
    
    centers = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        centers.append(pos)
    
    # All hexagons aligned at 30 degrees (flat-top orientation)
    inner = []
    for pos in centers:
        inner.append([float(pos[0]), float(pos[1]), 30.0])
    
    # Compute bounding side
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Refine angle
    for angle_deg in np.arange(best_angle - 2, best_angle + 2, 0.1):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Try multiple seed configurations
    seed_candidates = []
    
    # Seed 1: concentric ring
    seed1_inner, seed1_center, seed1_side, seed1_angle = _seed_solution()
    seed_candidates.append((seed1_inner, seed1_side, seed1_angle))
    
    # Seed 2: lattice packing
    seed2_inner, seed2_center, seed2_side, seed2_angle = _lattice_packing()
    seed_candidates.append((seed2_inner, seed2_side, seed2_angle))
    
    # Pick best seed
    best_seed = min(seed_candidates, key=lambda x: x[1])
    best_inner = [list(row) for row in best_seed[0]]
    best_side = best_seed[1]
    best_angle = best_seed[2]
    
    # Optimization: separate inner positions and outer angle
    # Phase 1: Optimize inner positions for fixed outer angle
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(best_seed[0]):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    
    # Try multiple outer angles
    outer_angles_to_try = [best_angle, 0.0, 15.0, 30.0, 45.0, 60.0]
    
    for outer_angle in outer_angles_to_try:
        if time.time() > DEADLINE - safety_margin:
            break
        
        try:
            # Use smooth surrogate for better Nelder-Mead convergence
            res = minimize(lambda p: _objective(p, outer_angle, n, smooth_beta=50.0),
                          init_params.copy(),
                          method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if time.time() > DEADLINE - safety_margin:
                break
            
            candidate_inner = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate_inner, outer_angle)
            
            # Check validity
            if _is_valid(candidate_inner, candidate_side, outer_angle, min_sep=1e-8):
                if candidate_side < best_side:
                    best_inner = candidate_inner.tolist()
                    best_side = candidate_side
                    best_angle = outer_angle
        except Exception:
            pass
    
    # Phase 2: Fine-tune outer angle with best inner positions
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        best_angle_fine = best_angle
        for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.5):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle_fine = angle_deg
        # Refine
        for angle_deg in np.arange(best_angle_fine - 0.5, best_angle_fine + 0.5, 0.05):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle_fine = angle_deg
        best_angle = best_angle_fine
    
    # Phase 3: One more round of inner optimization with refined angle (smooth)
    if time.time() < DEADLINE - safety_margin:
        try:
            init_params2 = np.zeros(3 * n)
            for i, (x, y, a) in enumerate(best_inner):
                init_params2[i*3] = x
                init_params2[i*3+1] = y
                init_params2[i*3+2] = a
            
            res2 = minimize(lambda p: _objective(p, best_angle, n, smooth_beta=100.0),
                           init_params2,
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if time.time() < DEADLINE - safety_margin:
                candidate_inner2 = res2.x.reshape(n, 3)
                candidate_side2 = _compute_side(candidate_inner2, best_angle)
                if _is_valid(candidate_inner2, candidate_side2, best_angle, min_sep=1e-8):
                    if candidate_side2 < best_side:
                        best_inner = candidate_inner2.tolist()
                        best_side = candidate_side2
        except Exception:
            pass
    
    # Phase 4: High-precision polish with exact (non-smooth) objective
    if time.time() < DEADLINE - safety_margin:
        try:
            init_params3 = np.zeros(3 * n)
            for i, (x, y, a) in enumerate(best_inner):
                init_params3[i*3] = x
                init_params3[i*3+1] = y
                init_params3[i*3+2] = a
            
            res3 = minimize(lambda p: _objective(p, best_angle, n, smooth_beta=0.0),
                           init_params3,
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if time.time() < DEADLINE - safety_margin:
                candidate_inner3 = res3.x.reshape(n, 3)
                candidate_side3 = _compute_side(candidate_inner3, best_angle)
                if _is_valid(candidate_inner3, candidate_side3, best_angle, min_sep=1e-9):
                    if candidate_side3 < best_side:
                        best_inner = candidate_inner3.tolist()
                        best_side = candidate_side3
        except Exception:
            pass
    
    # Final angle refinement (high precision)
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        for angle_deg in np.arange(best_angle - 2, best_angle + 2, 0.1):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
        for angle_deg in np.arange(best_angle - 0.2, best_angle + 0.2, 0.01):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
        for angle_deg in np.arange(best_angle - 0.02, best_angle + 0.02, 0.001):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
    
    # Final validation: recompute exact side from best_inner
    inner_arr_final = np.array(best_inner)
    final_side = _compute_side(inner_arr_final, best_angle, smooth_beta=0.0)
    
    # Verify validity one last time
    if not _is_valid(inner_arr_final, final_side, best_angle, min_sep=1e-9):
        # Fall back to seed if something went wrong
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        inner_rows = [[float(x), float(y), float(a)] for x, y, a in seed_inner]
        return inner_rows, (0.0, 0.0), float(seed_side), float(seed_angle)
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
    
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_angle)
    
    return inner_rows, center, side, angle_deg
