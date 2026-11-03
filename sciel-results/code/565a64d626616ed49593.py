"""Combined hexagon packing: kinematic hinge closure + LP outer optimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes`.
    Returns (min_side, center_x, center_y)
    """
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1
        b_vec[2*k] = Q_vals[k]
        
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1
        b_vec[2*k+1] = -P_vals[k]
    
    c_vec = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    return float('inf'), 0.0, 0.0

def _smooth_outer_side(hexes, oangle_rad, beta=200.0):
    """
    Differentiable approximation of the minimum outer hexagon side length.
    Uses soft-max to smooth the nonsmooth max over apothems.
    Returns (side, center_x, center_y) where side is the smoothed value.
    """
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    apothems = np.zeros(6)
    centers_proj = np.zeros(6)
    
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        apothems[k] = (np.max(projections) - np.min(projections)) / 2.0
        centers_proj[k] = (np.max(projections) + np.min(projections)) / 2.0
    
    # The side length is determined by the maximum apothem needed
    # Smooth this with soft-max
    beta_max = 200.0
    apothem_max = np.max(apothems)
    smooth_apothem = apothem_max + np.log(np.sum(np.exp(beta_max * (apothems - apothem_max)))) / beta_max
    
    # For center, use weighted average based on which faces are critical
    weights = np.exp(beta_max * (apothems - apothem_max))
    weights /= np.sum(weights)
    
    cx_est = 0.0
    cy_est = 0.0
    for k in range(6):
        ux, uy = u_vectors[k]
        cx_est += weights[k] * centers_proj[k] * ux
        cy_est += weights[k] * centers_proj[k] * uy
    
    side = smooth_apothem * 2.0 / math.sqrt(3)
    return side, cx_est, cy_est

def _build_hexagonal_lattice_with_hinges(hinge_angles, gap=1.001):
    """
    Build 11 unit hexagons in a hexagonal lattice topology with hinge deformation.
    
    The base topology is:
      - 1 center hexagon (index 0)
      - 6 ring-1 hexagons (indices 1-6) around the center
      - 4 selected ring-2 hexagons (indices 7-10)
    
    Hinge angles allow local rotation of ring-1 hexagons relative to center,
    and ring-2 hexagons relative to their ring-1 neighbors.
    
    hinge_angles: list of 10 values
      - indices 0-5: relative rotation of ring-1 hex k relative to center (radians)
      - indices 6-9: relative rotation of ring-2 hex k relative to its parent ring-1 hex (radians)
    
    Returns list of 11 (cx, cy, side, angle_deg) tuples.
    """
    sqrt3 = math.sqrt(3)
    
    # Base positions in axial coordinates
    # Center at (0,0)
    # Ring 1 at axial (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 selected: (2,0), (2,-1), (-2,-1), (-2,0)
    
    def axial_to_xy(q, r, scale=1.0):
        x = 1.5 * q * scale
        y = sqrt3 * (r + q / 2.0) * scale
        return (x, y)
    
    scale = gap
    
    # Center hexagon
    center_pos = (0.0, 0.0)
    center_angle = 0.0
    
    # Ring 1 base axial positions
    ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Compute ring-1 positions with hinge deformation
    ring1_positions = []
    ring1_angles = []
    for k in range(6):
        q, r = ring1_axial[k]
        base_x, base_y = axial_to_xy(q, r, scale)
        # Apply hinge rotation around center
        hinge = hinge_angles[k]
        cos_h = math.cos(hinge)
        sin_h = math.sin(hinge)
        # Rotate position around origin
        new_x = base_x * cos_h - base_y * sin_h
        new_y = base_x * sin_h + base_y * cos_h
        ring1_positions.append((new_x, new_y))
        ring1_angles.append(hinge * 180.0 / math.pi)
    
    # Ring 2: 4 selected hexagons
    # (2,0) parent is ring1[0]=(1,0)
    # (2,-1) parent is ring1[1]=(1,-1)
    # (-2,-1) parent is ring1[3]=(-1,0)
    # (-2,0) parent is ring1[2]=(0,-1) -- actually let's pick better parents
    
    # Let's use a simpler approach: place ring-2 hexagons relative to their nearest ring-1 neighbor
    ring2_specs = [
        # (axial_q, axial_r, parent_ring1_index, hinge_index)
        (2, 0, 0, 6),     # (2,0) relative to ring1[0]
        (2, -1, 1, 7),    # (2,-1) relative to ring1[1]
        (-2, -1, 3, 8),   # (-2,-1) relative to ring1[3]
        (-2, 0, 2, 9),    # (-2,0) relative to ring1[2]
    ]
    
    ring2_positions = []
    ring2_angles = []
    for q, r, parent_idx, hinge_idx in ring2_specs:
        # Base position
        base_x, base_y = axial_to_xy(q, r, scale)
        parent_x, parent_y = ring1_positions[parent_idx]
        
        # Vector from parent to base position
        vx = base_x - parent_x
        vy = base_y - parent_y
        
        # Apply hinge rotation around parent
        hinge = hinge_angles[hinge_idx]
        cos_h = math.cos(hinge)
        sin_h = math.sin(hinge)
        new_vx = vx * cos_h - vy * sin_h
        new_vy = vx * sin_h + vy * cos_h
        
        ring2_positions.append((parent_x + new_vx, parent_y + new_vy))
        ring2_angles.append(ring1_angles[parent_idx] + hinge * 180.0 / math.pi)
    
    # Assemble all hexagons
    hexes = []
    hexes.append((center_pos[0], center_pos[1], 1.0, center_angle))
    for k in range(6):
        hexes.append((ring1_positions[k][0], ring1_positions[k][1], 1.0, ring1_angles[k]))
    for k in range(4):
        hexes.append((ring2_positions[k][0], ring2_positions[k][1], 1.0, ring2_angles[k]))
    
    return hexes

def _objective_hinge(params):
    """
    Objective function for hinge optimization.
    params: [hinge_0, ..., hinge_9, outer_angle_rad]
    Returns the smooth outer side length.
    """
    hinge_angles = params[:10]
    outer_angle_rad = params[10]
    
    hexes = _build_hexagonal_lattice_with_hinges(hinge_angles, gap=1.001)
    
    # Check basic disjointness quickly (distance-based)
    for i in range(11):
        for j in range(i + 1, 11):
            dx = hexes[i][0] - hexes[j][0]
            dy = hexes[i][1] - hexes[j][1]
            if dx * dx + dy * dy < 3.0 * 0.99:  # (sqrt(3))^2 * 0.99
                return 100.0 + (3.0 - (dx * dx + dy * dy)) * 10
    
    side, _, _ = _smooth_outer_side(hexes, outer_angle_rad, beta=100.0)
    return side

def _optimize_hinges(max_iter=300, tol=1e-6):
    """
    Optimize hinge angles and outer orientation using Nelder-Mead.
    Returns (best_params, best_side, best_hexes, best_outer_angle).
    """
    # Initial parameters: all hinges zero, outer angle 0
    x0 = np.zeros(11)
    
    try:
        result = minimize(_objective_hinge, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        
        if result.fun < 100.0:  # Valid solution
            best_params = result.x
            hinge_angles = best_params[:10]
            outer_angle_rad = best_params[10]
            hexes = _build_hexagonal_lattice_with_hinges(hinge_angles, gap=1.001)
            side, ox, oy = _min_outer_side_analytical(hexes, outer_angle_rad)
            return best_params, side, hexes, outer_angle_rad * 180.0 / math.pi, (ox, oy)
    except Exception:
        pass
    
    return None, float('inf'), None, 0.0, (0.0, 0.0)

def _refine_outer_angle(hexes, init_angle_rad, tol=1e-5):
    """
    Refine the outer hexagon angle using golden section search.
    Returns (best_angle_rad, best_side, best_center).
    """
    lo = max(0.0, init_angle_rad - 0.5)
    hi = min(math.pi / 6, init_angle_rad + 0.5)  # 30 degrees
    
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    
    fc = _min_outer_side_analytical(hexes, c)[0]
    fd = _min_outer_side_analytical(hexes, d)[0]
    
    for _ in range(25):
        if time.time() > _DEADLINE - 5.0:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = _min_outer_side_analytical(hexes, c)[0]
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = _min_outer_side_analytical(hexes, d)[0]
    
    best_angle = (lo + hi) / 2.0
    side, ox, oy = _min_outer_side_analytical(hexes, best_angle)
    return best_angle, side, (ox, oy)

def construct():
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        sqrt3 = math.sqrt(3)
        inner = []
        for k in range(N):
            q = k % 7
            r = k // 7
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Phase 1: Optimize hinge angles
    if time.time() < _DEADLINE - 40.0:
        params, side, hexes, outer_angle_deg, center = _optimize_hinges(max_iter=200, tol=1e-5)
        
        if hexes is not None and side < best_side:
            # Refine outer angle
            if time.time() < _DEADLINE - 15.0:
                best_angle_rad, refined_side, refined_center = _refine_outer_angle(
                    hexes, math.radians(outer_angle_deg), tol=1e-5)
                if refined_side < side:
                    side = refined_side
                    outer_angle_deg = best_angle_rad * 180.0 / math.pi
                    center = refined_center
            
            if side < best_side:
                best_side = side
                best_inner = [[h[0], h[1], h[3]] for h in hexes]
                best_center = center
                best_outer_angle = outer_angle_deg
    
    # Phase 2: Try the standard hexagonal lattice as fallback/comparison
    if time.time() < _DEADLINE - 20.0:
        sqrt3 = math.sqrt(3)
        GAP = 1.001
        
        def axial_to_xy(q, r):
            x = 1.5 * q * GAP
            y = sqrt3 * (r + q / 2.0) * GAP
            return (x, y)
        
        pos_axial = [(0, 0)]
        pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
        ring2_candidates = [
            (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
            (0, 2), (1, 2)
        ]
        
        all_axial = pos_axial + ring2_candidates
        all_xy = [axial_to_xy(q, r) for q, r in all_axial]
        ring2_indices = list(range(7, len(all_xy)))
        
        # Try a few combinations of ring-2 hexagons
        for combo in itertools.combinations(ring2_indices, 4):
            if time.time() > _DEADLINE - 15.0:
                break
            
            selected_indices = list(range(7)) + list(combo)
            full_xy = [all_xy[i] for i in selected_indices]
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            
            # Quick coarse angle search
            best_coarse_angle = 0.0
            best_coarse_side = float('inf')
            for ang_deg in np.linspace(0, 30, 16):
                side_c, _, _ = _smooth_outer_side(inner_hexes, math.radians(ang_deg), beta=100.0)
                if side_c < best_coarse_side:
                    best_coarse_side = side_c
                    best_coarse_angle = ang_deg
            
            # Refine
            if time.time() < _DEADLINE - 10.0:
                refined_angle, refined_side, refined_center = _refine_outer_angle(
                    inner_hexes, math.radians(best_coarse_angle), tol=1e-5)
                
                if refined_side < best_side:
                    best_side = refined_side
                    best_inner = [[x, y, 0.0] for x, y in full_xy]
                    best_center = refined_center
                    best_outer_angle = refined_angle * 180.0 / math.pi
    
    # Fallback if nothing worked
    if best_inner is None:
        sqrt3 = math.sqrt(3)
        GAP = 1.001
        pos_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (-2, -1), (-2, 0)]
        best_inner = []
        for q, r in pos_axial:
            x = 1.5 * q * GAP
            y = sqrt3 * (r + q / 2.0) * GAP
            best_inner.append([x, y, 0.0])
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y, _ in best_inner]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes, 0.0)
        best_center = (ox, oy)
        best_outer_angle = 0.0
    
    # Validate disjointness
    inner_hexes = [(h[0], h[1], 1.0, h[2]) for h in best_inner]
    if not _all_disjoint(inner_hexes):
        # Try increasing gap
        for extra_gap in [1.002, 1.005, 1.01, 1.02]:
            scale = extra_gap / 1.001
            inner_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(inner_hexes):
                best_inner = [[h[0], h[1], h[3]] for h in inner_hexes]
                # Recompute outer
                best_side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_outer_angle))
                best_center = (ox, oy)
                break
    
    # Validate containment
    outer_hex = (best_center[0], best_center[1], best_side, best_outer_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        best_side *= 1.01
        outer_hex = (best_center[0], best_center[1], best_side, best_outer_angle)
        
        valid = True
        for h in inner_hexes:
            if not _inner_in_outer(h, outer_hex):
                valid = False
                break

    return best_inner, (best_center[0], best_center[1]), best_side, best_outer_angle
