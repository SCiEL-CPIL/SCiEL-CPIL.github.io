import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

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
    that contains all hexagons in `hexes` (centered at origin).
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None

    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    ring2_indices = list(range(7, len(all_xy)))

    # 1. Find best lattice configuration
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline:
            break
            
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Ternary search for best angle
        lo = 0.0
        hi = 30.0
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        
        fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        for _ in range(15):
            if time.time() > deadline:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle_candidate))
        
        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_center = (ox + cx_mean, oy + cy_mean)
            best_xy = full_xy
            # Store the centered hexes for polishing
            best_inner_hexes = [(x + cx_mean, y + cy_mean, 1.0, 0.0) for x, y in centered_xy]

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        cx_mean = sum(x for x, y in best_xy) / 7.0
        cy_mean = sum(y for x, y in best_xy) / 7.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0
        best_inner_hexes = [(x + cx_mean, y + cy_mean, 1.0, 0.0) for x, y in centered_xy]

    # 2. Polishing Step: Local Search on positions and angles
    if time.time() < deadline - 20.0 and best_inner_hexes:
        # Variables: x, y, theta for each hexagon
        x0 = []
        for h in best_inner_hexes:
            x0.extend([h[0], h[1], h[3]])
        
        def cost_func(params):
            # Unpack
            hexes = []
            for i in range(11):
                x = params[3*i]
                y = params[3*i+1]
                theta = params[3*i+2]
                hexes.append((x, y, 1.0, theta))
            
            # Check overlaps
            overlap_penalty = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    if not _hexagons_disjoint(hexes[i], hexes[j]):
                        # Simple penalty: distance between centers
                        dx = hexes[i][0] - hexes[j][0]
                        dy = hexes[i][1] - hexes[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        overlap_penalty += (1.7 - dist)**2 if dist < 1.7 else 0
                        # Hard penalty if significant overlap
                        if dist < 1.0:
                            overlap_penalty += 100.0
                            
            if overlap_penalty > 1e-6:
                # Return large cost
                return 1000.0 + overlap_penalty
            
            # Get min outer side
            # Center the hexes for calculation? No, the LP finds the best center.
            # But the LP expects the hexes to be in absolute coords.
            side, ox, oy = _min_outer_side_analytical(hexes, math.radians(best_angle))
            
            return side + overlap_penalty

        try:
            # Use Nelder-Mead
            result = minimize(cost_func, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_side:
                # Update best
                best_side = result.fun
                # Extract new positions
                new_hexes = []
                for i in range(11):
                    x = result.x[3*i]
                    y = result.x[3*i+1]
                    theta = result.x[3*i+2]
                    new_hexes.append((x, y, 1.0, theta))
                
                # Recalculate outer hexagon for final answer
                side, ox, oy = _min_outer_side_analytical(new_hexes, math.radians(best_angle))
                if side < best_side:
                    best_side = side
                    best_inner_hexes = new_hexes
                    best_center = (ox, oy)
        except Exception:
            pass

    # Finalize
    inner_hexes = best_inner_hexes
    if not _all_disjoint(inner_hexes):
        # Safety: revert to lattice if polishing broke it
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y in inner_hexes) / 11.0
        cy_mean = sum(y for x, y in inner_hexes) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (best_center[0], best_center[1]), best_side, best_angle
