import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
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

def _min_outer_side_lp(vertices, angle_deg):
    """
    Exact LP solution for minimum enclosing hexagon side length.
    Returns (center_x, center_y, side_length) or None.
    """
    if not vertices:
        return 0.0, 0.0, 0.0
    
    pts = np.array(vertices)
    
    normals = np.array([
        [math.cos(math.radians(angle_deg + 30 + 60 * k)), 
         math.sin(math.radians(angle_deg + 30 + 60 * k))]
        for k in range(6)
    ])
    
    proj = pts @ normals.T
    M = proj.max(axis=0)
    m = proj.min(axis=0)
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        # Constraint: n_k . c + t >= M_k  => -n_k . c - t <= -M_k
        A_ub.append([-nx, -ny, -1.0])
        b_ub.append(-M[k])
        # Constraint: n_k . c - t <= m_k  => n_k . c - t <= m_k
        A_ub.append([nx, ny, -1.0])
        b_ub.append(m[k])
    
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    c_vec = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_vec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    
    return None

def _min_outer_side_smooth(vertices, angle_deg, beta=50.0):
    """
    Smooth approximation of the minimum enclosing hexagon side length using log-sum-exp.
    This allows for gradient-based or Nelder-Mead optimization without LP overhead.
    Returns estimated side length.
    """
    if not vertices:
        return 0.0
    
    pts = np.array(vertices)
    
    normals = np.array([
        [math.cos(math.radians(angle_deg + 30 + 60 * k)), 
         math.sin(math.radians(angle_deg + 30 + 60 * k))]
        for k in range(6)
    ])
    
    proj = pts @ normals.T
    M = proj.max(axis=0)
    m = proj.min(axis=0)
    
    # Widths along each normal direction
    widths = M - m
    
    # The side length S is related to the apothem A by S = 2A/sqrt(3).
    # The constraint is that for each direction k, the projection width must be <= 2 * A_k (where A_k is the apothem in that direction).
    # Actually, for a regular hexagon with side S, the distance from center to side (apothem) is S * sqrt(3) / 2.
    # The condition for containment is:
    # max_proj_k - min_proj_k <= 2 * (S * sqrt(3) / 2) = S * sqrt(3)
    # So S >= width_k / sqrt(3) for all k.
    # We want min S such that S >= max_k (width_k / sqrt(3)).
    
    # To make it smooth and allow center optimization, we usually fix the center to the centroid or optimize it.
    # Here, we assume the center is the centroid of the points for the smooth estimate, 
    # or we can optimize center separately. For speed in the coarse search, 
    # let's just compute the required S based on widths, assuming optimal centering (centroid).
    # Note: The optimal center for a hexagon is not necessarily the centroid, 
    # but for a smooth surrogate, it's a good start.
    
    # Actually, a better smooth objective is to minimize the maximum "violation" 
    # or use a soft max of the required side lengths.
    
    required_sides = widths / math.sqrt(3)
    
    # Soft max to approximate max(required_sides)
    # log-sum-exp: log(sum(exp(x/beta))) * beta
    max_val = np.max(required_sides)
    if max_val < -1000:
        return 0.0
    shifted = required_sides - max_val
    sum_exp = np.sum(np.exp(shifted / beta))
    soft_max = max_val + beta * math.log(sum_exp)
    
    return soft_max

def _find_best_for_positions(positions, inner_angle, deadline):
    """
    Given fixed inner hexagon positions and angle, find best outer hexagon.
    Returns (best_side, best_angle, best_center)
    """
    all_vertices = []
    for px, py in positions:
        verts = _hex_vertices(px, py, 1.0, inner_angle)
        all_vertices.extend(verts)
    
    if not all_vertices:
        return float('inf'), 0.0, (0.0, 0.0)
    
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse scan: 0 to 30 degrees
    coarse_angles = np.linspace(0, 30, 61)
    
    for ang_deg in coarse_angles:
        if time.time() > deadline:
            break
        # Use LP for accuracy in coarse phase, it's fast enough
        res = _min_outer_side_lp(all_vertices, ang_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = ang_deg
                best_center = (cx, cy)
    
    # Fine scan around best coarse angle
    if time.time() < deadline - 5.0:
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine_angles = np.linspace(fine_lo, fine_hi, 21)
        
        for ang_deg in fine_angles:
            if time.time() > deadline:
                break
            res = _min_outer_side_lp(all_vertices, ang_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_angle = ang_deg
                    best_center = (cx, cy)
    
    # Nelder-Mead polish on the angle using LP objective
    if time.time() < deadline - 3.0:
        def objective(angle_arr):
            ang = angle_arr[0] % 30.0
            if ang > 15.0:
                ang = 30.0 - ang
            res = _min_outer_side_lp(all_vertices, ang)
            if res is None:
                return 1e10
            return res[2]
        
        x0 = np.array([best_angle])
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8,
                                    'adaptive': True})
            if result.fun < best_side:
                ang = result.x[0] % 30.0
                if ang > 15.0:
                    ang = 30.0 - ang
                res = _min_outer_side_lp(all_vertices, ang)
                if res:
                    best_side = res[2]
                    best_angle = ang
                    best_center = (res[0], res[1])
        except Exception:
            pass
        
        # Final polish
        if time.time() < deadline - 1.0:
            try:
                result2 = minimize(objective, np.array([best_angle]),
                                method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-10,
                                        'adaptive': True})
                if result2.fun < best_side:
                    ang = result2.x[0] % 30.0
                    if ang > 15.0:
                        ang = 30.0 - ang
                    res = _min_outer_side_lp(all_vertices, ang)
                    if res:
                        best_side = res[2]
                        best_angle = ang
                        best_center = (res[0], res[1])
            except Exception:
                pass
    
    return best_side, best_angle, best_center

def construct():
    if time.time() > _DEADLINE - 10.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    GAP = 1.0001
    
    def axial_to_xy(q, r):
        x = 1.5 * q * GAP
        y = sqrt3 * (r + q / 2.0) * GAP
        return (x, y)
    
    # Core 7 hexagons
    pos_axial_core = [(0, 0)]
    pos_axial_core.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # Candidates for 4 more from second ring
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    core_xy = [axial_to_xy(q, r) for q, r in pos_axial_core]
    ring2_xy = [axial_to_xy(q, r) for q, r in ring2_candidates]
    
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_positions = None
    best_inner_angle = 0.0
    
    # Try different inner rotation angles
    # Due to symmetry of the hexagon, rotating by 30 degrees is equivalent to a different orientation.
    # We try 0 and 30.
    inner_angles_to_try = [0.0, 30.0]
    
    for inner_angle in inner_angles_to_try:
        if time.time() > _DEADLINE - 15.0:
            break
        
        for combo in itertools.combinations(range(len(ring2_xy)), 4):
            if time.time() > _DEADLINE - 10.0:
                break
            
            selected_indices = list(range(7)) + list(combo)
            all_xy = core_xy + [ring2_xy[i] for i in combo]
            
            # Verify disjointness
            inner_hexes = [(x, y, 1.0, inner_angle) for x, y in all_xy]
            if not _all_disjoint(inner_hexes):
                continue
            
            # Find best outer hexagon
            side, ang, center = _find_best_for_positions(all_xy, inner_angle, _DEADLINE - 5.0)
            
            if side < best_side:
                best_side = side
                best_angle = ang
                best_center = center
                best_positions = all_xy
                best_inner_angle = inner_angle
    
    if best_positions is None:
        best_positions = core_xy
        best_inner_angle = 0.0
        side, ang, center = _find_best_for_positions(best_positions, 0.0, _DEADLINE - 5.0)
        if side != float('inf'):
            best_side = side
            best_angle = ang
            best_center = center
        else:
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final verification
    inner_hexes = [(x, y, 1.0, best_inner_angle) for x, y in best_positions]
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        best_side *= 1.01
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    
    return inner, (best_center[0], best_center[1]), best_side, best_angle
