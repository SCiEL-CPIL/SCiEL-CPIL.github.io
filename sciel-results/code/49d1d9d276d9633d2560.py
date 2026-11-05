import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _min_outer_side(inner_params, outer_angle_deg):
    """
    Compute minimum side length of outer hexagon containing all inner hexagons.
    Outer hexagon centered at origin with given rotation.
    """
    # Normals for the outer hexagon sides
    # The sides are perpendicular to the normals.
    # For a regular hexagon, the normals are at angles outer_angle + 30, outer_angle + 90, etc?
    # Let's define the hexagon by its support function.
    # A regular hexagon with side S centered at origin, rotated by theta.
    # The distance from center to each side (apothem) is A = S * sqrt(3) / 2.
    # The normals to the sides are at angles theta + 30 + k*60.
    normals = np.array([
        [np.cos(np.deg2rad(outer_angle_deg + 30 + k * 60)),
         np.sin(np.deg2rad(outer_angle_deg + 30 + k * 60))]
        for k in range(6)
    ])
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(inner_params)):
        x, y, a = inner_params[i]
        verts = _hex_vertices(x, y, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For each normal direction, find the max and min projection
    # The width in that direction is max_proj - min_proj
    # The required apothem A must be >= width / 2
    # So A >= max_k (width_k / 2)
    # Side S = 2 * A / sqrt(3)
    
    max_width = 0.0
    for n_vec in normals:
        projs = all_verts @ n_vec
        width = np.max(projs) - np.min(projs)
        if width > max_width:
            max_width = width
            
    A = max_width / 2.0
    side = 2.0 * A / SQRT3
    return side

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

def _pairwise_min_dist(v1, v2):
    """Minimum distance between two convex hexagons (6 vertices each)."""
    # Check if any vertex of one is inside the other
    for v in v1:
        if _point_in_polygon(v, v2):
            return 0.0
    for v in v2:
        if _point_in_polygon(v, v1):
            return 0.0
    # Compute min distance between all edge pairs
    min_d = float('inf')
    for i in range(6):
        a1 = v1[i]
        a2 = v1[(i+1) % 6]
        for j in range(6):
            b1 = v2[j]
            b2 = v2[(j+1) % 6]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_d:
                min_d = d
    return min_d

def _overlap_penalty(inner_params, weight=1000.0, threshold=0.01):
    """
    Fast overlap penalty using center distances as a proxy,
    with exact check only for close pairs.
    """
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = inner_params[j, 0] - inner_params[i, 0]
            dy = inner_params[j, 1] - inner_params[i, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Max distance for non-overlap is 2.0 (vertex to vertex of two hexes touching at a point? No, center to center max is 2.0 if they touch at vertices? 
            # Actually, if two hexes are rotated 30 deg relative, they can be closer.
            # Safe bound: if dist < 2.0, check.
            if dist < 2.0:
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _pairwise_min_dist(v1, v2)
                if sep < threshold:
                    penalty += weight * (threshold - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    """Objective: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _min_outer_side(inner_params, outer_angle_deg)
    pen = _overlap_penalty(inner_params)
    return side + pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=1e-6):
    """Validate configuration."""
    # Check side length consistency
    calc_side = _min_outer_side(inner_params, outer_angle_deg)
    if calc_side > side + 1e-5:
        return False
    
    n = len(inner_params)
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _pairwise_min_dist(v1, v2)
            if sep < min_sep:
                return False
    return True

def _generate_valid_seed():
    """
    Generate a valid non-overlapping seed using a hexagonal lattice.
    """
    centers = []
    angles = []
    
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    
    # Ring 1: 6 hexes
    # Distance sqrt(3)
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = SQRT3 * np.cos(ang)
        y = SQRT3 * np.sin(ang)
        centers.append((x, y))
        angles.append(0.0)
        
    # Ring 2: 4 hexes
    # Distance 2*sqrt(3)
    # Pick 4 symmetric positions: 30, 90, 150, 210? No, let's pick 0, 60, 120, 180?
    # If we pick 0, 60, 120, 180, they are at (2*sqrt(3), 0), (sqrt(3), 3), (-sqrt(3), 3), (-2*sqrt(3), 0).
    # These are further out.
    # Let's try 30, 90, 150, 210, 270, 330?
    # Distance 2*sqrt(3) at angle 30: x = 2*sqrt(3)*cos(30) = 3, y = 2*sqrt(3)*sin(30) = sqrt(3).
    # Point (3, sqrt(3)).
    # Distance to Ring 1 hex at (sqrt(3), 0) is sqrt((3-sqrt(3))^2 + (sqrt(3))^2) = sqrt(9 - 6sqrt(3) + 3 + 3) = sqrt(15 - 6sqrt(3)) ~ sqrt(4.6) ~ 2.15.
    # This is > 2.0, so no overlap risk even with rotation.
    
    for ang_deg in [30, 90, 150, 210, 270, 330]:
        # We only need 4. Let's take 30, 150, 210, 330 (top, bottom-left, bottom-right, bottom? No, 30 is top-rightish)
        # 30: (3, sqrt(3))
        # 90: (0, 2*sqrt(3))
        # 150: (-3, sqrt(3))
        # 210: (-3, -sqrt(3))
        # 270: (0, -2*sqrt(3))
        # 330: (3, -sqrt(3))
        # Let's pick 90, 270 (top, bottom) and 30, 210 (right-top, left-bottom)?
        # Or just 0, 60, 120, 180?
        # Let's use 30, 90, 150, 210.
        pass
        
    # Let's just use 4 specific ones:
    # 90 deg: (0, 2*sqrt(3))
    # 270 deg: (0, -2*sqrt(3))
    # 30 deg: (3, sqrt(3))
    # 210 deg: (-3, -sqrt(3))
    
    for ang_deg in [90, 270, 30, 210]:
        ang = np.deg2rad(ang_deg)
        x = 2 * SQRT3 * np.cos(ang)
        y = 2 * SQRT3 * np.sin(ang)
        centers.append((x, y))
        angles.append(0.0)
        
    inner = []
    for (x, y), a in zip(centers, angles):
        inner.append([x, y, a])
        
    return np.array(inner)

def construct():
    start_time = time.time()
    DEADLINE = 145.0 # Safety margin for 150s limit
    
    n = 11
    outer_angle = 0.0 # Start with 0
    
    # 1. Get valid seed
    seed_inner = _generate_valid_seed()
    best_inner = seed_inner.copy()
    best_side = _min_outer_side(best_inner, outer_angle)
    
    # 2. Optimization
    # We will try to optimize the inner positions and rotations.
    # We will also try a few outer angles.
    
    best_overall_side = best_side
    best_overall_inner = best_inner.copy()
    best_overall_angle = outer_angle
    
    # Try a few outer angles
    candidate_angles = [0.0, 30.0, 60.0]
    
    for o_ang in candidate_angles:
        if time.time() > DEADLINE - 20:
            break
            
        current_side = _min_outer_side(best_inner, o_ang)
        
        # If this angle is better, update best
        if current_side < best_overall_side:
            best_overall_side = current_side
            best_overall_inner = best_inner.copy()
            best_overall_angle = o_ang
            
        # Optimize inner params for this outer angle
        # Use Nelder-Mead
        x0 = best_inner.flatten()
        
        try:
            res = minimize(_objective, x0, args=(o_ang, n),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_overall_side + 0.1: # Only accept if promising
                candidate_inner = res.x.reshape(n, 3)
                candidate_side = _min_outer_side(candidate_inner, o_ang)
                
                # Check validity
                if _is_valid(candidate_inner, candidate_side, o_ang, min_sep=1e-5):
                    if candidate_side < best_overall_side:
                        best_overall_side = candidate_side
                        best_overall_inner = candidate_inner
                        best_overall_angle = o_ang
        except Exception:
            pass
            
    # 3. Final Polish
    if time.time() < DEADLINE - 10:
        # Try to polish the best found configuration
        x0 = best_overall_inner.flatten()
        try:
            res = minimize(_objective, x0, args=(best_overall_angle, n),
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            candidate_inner = res.x.reshape(n, 3)
            candidate_side = _min_outer_side(candidate_inner, best_overall_angle)
            
            if _is_valid(candidate_inner, candidate_side, best_overall_angle, min_sep=1e-5):
                if candidate_side < best_overall_side:
                    best_overall_side = candidate_side
                    best_overall_inner = candidate_inner
        except Exception:
            pass
            
    # 4. Final Validation and Fallback
    final_side = _min_outer_side(best_overall_inner, best_overall_angle)
    if not _is_valid(best_overall_inner, final_side, best_overall_angle, min_sep=1e-5):
        # Fallback to seed
        best_overall_inner = seed_inner
        best_overall_angle = 0.0
        final_side = _min_outer_side(best_overall_inner, best_overall_angle)
        
    # Format output
    inner_rows = []
    for row in best_overall_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_overall_angle)
    
    return inner_rows, center, side, angle_deg
