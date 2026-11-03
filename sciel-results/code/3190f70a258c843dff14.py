import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s hard limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _check_disjoint(inner_positions, inner_angles, tol=1e-9):
    """
    Check that all inner hexagons are disjoint (no overlap).
    inner_positions: list of (x, y)
    inner_angles: list of angle_degrees
    Returns True if all pairs are disjoint.
    """
    n = len(inner_positions)
    # Precompute vertices for each hexagon
    all_verts = []
    for i in range(n):
        px, py = inner_positions[i]
        ang = inner_angles[i]
        all_verts.append(_get_hex_vertices(px, py, ang))
    
    # For each pair, check if any edge of hex i intersects any edge of hex j
    # Or if any vertex of one is inside the other
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(all_verts[i], all_verts[j]):
                return False
    return True

def _hexagons_overlap(verts1, verts2):
    """Check if two convex hexagons overlap (interior intersection)."""
    # Check if any vertex of hex1 is strictly inside hex2 or vice versa
    if _point_in_convex_polygon(verts1[0], verts2, strict=True):
        return True
    if _point_in_convex_polygon(verts2[0], verts1, strict=True):
        return True
    
    # Check edge intersections
    for k in range(6):
        a1 = verts1[k]
        b1 = verts1[(k + 1) % 6]
        for l in range(6):
            a2 = verts2[l]
            b2 = verts2[(l + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _point_in_convex_polygon(p, polygon, strict=False):
    """Check if point p is inside convex polygon. strict=True means strictly inside."""
    px, py = p
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if strict:
            if cross < 0:
                return False
        else:
            if cross < -1e-12:
                return False
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0:
            if sign > 0:
                return False
            sign = -1
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (including endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    
    # Check collinear cases
    if abs(c1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    
    return False

def _on_segment(a, b, p):
    """Check if point p is on segment a-b."""
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)

def _find_min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices, using LP.
    Returns (center_x, center_y, side_length) or None.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # (c - v) . n <= R  =>  nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
            # -(c - v) . n <= R  =>  -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, 
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _compute_all_vertices(positions, angles):
    """Compute all 6*N vertices for given positions and angles."""
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        all_verts.extend(_get_hex_vertices(px, py, ang))
    return all_verts

def _nelder_mead_polish(positions, angles, max_iter=200, tol=1e-6):
    """
    Polish the inner positions using Nelder-Mead with fast LP evaluation.
    Returns improved positions and angles.
    """
    n = len(positions)
    
    # Flatten parameters: x0, y0, x1, y1, ..., angle0, angle1, ...
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    # Precompute vertex offsets for unit hexagon at angle 0
    base_verts = np.array([(math.cos(math.pi * k / 3), math.sin(math.pi * k / 3)) for k in range(6)])
    
    def fast_disjoint_check(pos, ang):
        """Fast pairwise distance check for disjointness."""
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Minimum distance between centers of two unit hexagons when disjoint
                # is at least sqrt(3) * 2 / 2 = sqrt(3) for same orientation
                # For different orientations, minimum can be smaller, use conservative bound
                if dist < 1.732 * 0.99:  # sqrt(3) * 0.99
                    # Do precise check
                    if _check_disjoint([pos[i], pos[j]], [ang[i], ang[j]]):
                        return False
        return True
    
    def objective(params):
        if time.time() > _DEADLINE - 20.0:
            return 1e8
        
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Fast disjointness check
        if not fast_disjoint_check(pos, ang):
            return 1e6
        
        all_verts = _compute_all_vertices(pos, ang)
        
        # Find best outer angle with coarse search
        best_side = float('inf')
        for outer_angle in np.arange(0, 30.1, 3.0):
            res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
        
        return best_side
    
    # Use Nelder-Mead with tighter settings
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol,
                               'adaptive': True, 'initial_simplex': None})
        if res.success:
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            # Verify final result
            if _check_disjoint(new_pos, new_ang):
                return new_pos, new_ang
    except:
        pass
    
    return positions, angles

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    # Start with hexagonal cluster arrangement
    # Center hexagon at origin, 6 around it, then 6 more in second ring
    scale = 1.000001  # Slightly larger than touching
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate positions on triangular lattice
    # Hexagonal cluster: center + 6 neighbors + 6 more
    lattice_positions = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            lattice_positions.append((x, y))
    
    # Sort by distance from origin and take N closest
    lattice_positions.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    initial_positions = lattice_positions[:N]
    initial_angles = [0.0] * N
    
    # Also try a compact arrangement: center + ring of 6 + partial second ring
    compact_positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.pi * k / 3
        compact_positions.append((sqrt3 * scale * math.cos(a), sqrt3 * scale * math.sin(a)))
    # Add 4 more from second ring at optimal angles
    for k in [0, 1, 2, 3]:
        a = math.pi * k / 3 + math.pi / 6
        r = 2 * sqrt3 * scale
        compact_positions.append((r * math.cos(a), r * math.sin(a)))
    compact_positions = compact_positions[:N]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try different inner angle configurations
    angle_configs = [
        [0.0] * N,
        [30.0] * N,
    ]
    
    # Multiple initial position sets to try
    position_sets = [initial_positions, compact_positions]
    
    for pos_set in position_sets:
        for inner_angles in angle_configs:
            if time.time() > _DEADLINE - 15.0:
                break
            
            positions = list(pos_set)
            angles = inner_angles
        
        # First pass: find best outer angle for initial configuration
        all_verts = _compute_all_vertices(positions, angles)
        
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (0.0, 0.0)
        
        # Coarse search for outer angle
        for outer_angle_deg in np.arange(0, 30.1, 1.0):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
        
        # Fine search around best coarse angle
        if best_coarse_angle > 0:
            for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0), 
                                            min(30, best_coarse_angle + 1.0) + 0.01, 0.05):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
        
        if best_coarse_side < best_side:
            best_side = best_coarse_side
            best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
            best_center = best_coarse_center
            best_outer_angle = best_coarse_angle
        
        # Nelder-Mead polish
        if time.time() < _DEADLINE - 30.0:
            polished_pos, polished_ang = _nelder_mead_polish(positions, angles, max_iter=150)
            
            if polished_pos != positions or polished_ang != angles:
                # Verify disjointness
                if _check_disjoint(polished_pos, polished_ang):
                    all_verts_polished = _compute_all_vertices(polished_pos, polished_ang)
                    
                    # Find best outer angle for polished configuration
                    best_polished_side = float('inf')
                    best_polished_angle = 0.0
                    best_polished_center = (0.0, 0.0)
                    
                    for outer_angle_deg in np.arange(0, 30.1, 0.5):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _find_min_enclosing_hex_side_lp(all_verts_polished, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_polished_side:
                                best_polished_side = side
                                best_polished_angle = outer_angle_deg
                                best_polished_center = (cx, cy)
                    
                    # Fine polish on outer angle
                    if best_polished_angle > 0:
                        for outer_angle_deg in np.arange(max(0, best_polished_angle - 0.5), 
                                                        min(30, best_polished_angle + 0.5) + 0.01, 0.01):
                            if time.time() > _DEADLINE - 15.0:
                                break
                            res = _find_min_enclosing_hex_side_lp(all_verts_polished, outer_angle_deg)
                            if res:
                                cx, cy, side = res
                                if side < best_polished_side:
                                    best_polished_side = side
                                    best_polished_angle = outer_angle_deg
                                    best_polished_center = (cx, cy)
                    
                    if best_polished_side < best_side:
                        best_side = best_polished_side
                        best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(polished_pos, polished_ang)]
                        best_center = best_polished_center
                        best_outer_angle = best_polished_angle
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final high-precision polish: refine outer angle with very fine steps
    if time.time() < _DEADLINE - 5.0 and best_inner is not None:
        all_verts_final = _compute_all_vertices(
            [(p[0], p[1]) for p in best_inner],
            [p[2] for p in best_inner]
        )
        
        # Very fine search around best angle
        final_best_side = best_side
        final_best_angle = best_outer_angle
        final_best_center = best_center
        
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 0.2), 
                                        min(30, best_outer_angle + 0.2) + 0.001, 0.002):
            if time.time() > _DEADLINE - 3.0:
                break
            res = _find_min_enclosing_hex_side_lp(all_verts_final, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < final_best_side:
                    final_best_side = side
                    final_best_angle = outer_angle_deg
                    final_best_center = (cx, cy)
        
        best_side = final_best_side
        best_center = final_best_center
        best_outer_angle = final_best_angle
    
    return best_inner, best_center, best_side, best_outer_angle
