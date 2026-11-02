"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern with LP-based outer hexagon optimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _optimize_outer_lp(inner_hexes, initial_cx, initial_cy, initial_angle, deadline):
    """Optimize outer hexagon using linear programming for center at each angle,
    combined with fine angle sweep."""
    sqrt3 = math.sqrt(3)
    best_side = float('inf')
    best_cx = initial_cx
    best_cy = initial_cy
    best_angle = initial_angle
    
    # Precompute all vertices
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    n_verts = len(all_verts)
    
    def eval_side_at_angle(angle_deg, cx=None, cy=None):
        """Evaluate side length at given angle, using LP to find optimal center if cx/cy is None."""
        a = -math.radians(angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        
        # Direction vectors for 6 normals
        dir_vectors = []
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dir_vectors.append((math.cos(theta), math.sin(theta)))
        
        # For each vertex and each direction, compute:
        # proj = (vx - cx) * nx + (vy - cy) * ny
        #      = vx*nx + vy*ny - cx*nx - cy*ny
        # We want to minimize max over all (v, d) of proj
        # This is equivalent to: minimize t such that for all (v,d):
        #   vx*nx + vy*ny - cx*nx - cy*ny <= t
        
        # LP variables: cx, cy, t
        # Minimize t
        # Constraints: for each (v, d): -nx*cx - ny*cy + t >= vx*nx + vy*ny
        #              i.e., nx*cx + ny*cy - t <= -(vx*nx + vy*ny)
        
        if cx is None or cy is None:
            # Use LP
            c = np.array([0.0, 0.0, 1.0])  # minimize t
            
            A_ub = []
            b_ub = []
            for vx, vy in all_verts:
                for nx, ny in dir_vectors:
                    # nx*cx + ny*cy - t <= -(vx*nx + vy*ny)
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(-(vx * nx + vy * ny))
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Bounds: cx, cy in reasonable range, t >= 0
            bounds = [(-10, 10), (-10, 10), (0, None)]
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                opt_cx = result.x[0]
                opt_cy = result.x[1]
                t = result.x[2]
                side = t * 2.0 / sqrt3
                return side, opt_cx, opt_cy
            else:
                # Fallback to centroid
                cx_fallback = sum(v[0] for v in all_verts) / n_verts
                cy_fallback = sum(v[1] for v in all_verts) / n_verts
                max_val = 0.0
                for vx, vy in all_verts:
                    dx = vx - cx_fallback
                    dy = vy - cy_fallback
                    lx = dx * cos_a - dy * sin_a
                    ly = dx * sin_a + dy * cos_a
                    for nx, ny in dir_vectors:
                        proj = lx * nx + ly * ny
                        if proj > max_val:
                            max_val = proj
                side = max_val * 2.0 / sqrt3
                return side, cx_fallback, cy_fallback
        else:
            # Just evaluate at given center
            max_val = 0.0
            for vx, vy in all_verts:
                dx = vx - cx
                dy = vy - cy
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                for nx, ny in dir_vectors:
                    proj = lx * nx + ly * ny
                    if proj > max_val:
                        max_val = proj
            side = max_val * 2.0 / sqrt3
            return side, cx, cy
    
    # Coarse angle sweep with LP center optimization
    angles_coarse = list(np.arange(0, 60, 1.0))
    for ang in angles_coarse:
        if time.time() > deadline - 20:
            break
        side, cx, cy = eval_side_at_angle(ang)
        if side < best_side:
            best_side = side
            best_cx = cx
            best_cy = cy
            best_angle = ang
    
    # Fine angle sweep around best
    if time.time() < deadline - 10:
        angles_fine = list(np.arange(best_angle - 3, best_angle + 3, 0.1))
        for ang in angles_fine:
            if time.time() > deadline - 10:
                break
            side, cx, cy = eval_side_at_angle(ang)
            if side < best_side:
                best_side = side
                best_cx = cx
                best_cy = cy
                best_angle = ang
    
    # Very fine sweep
    if time.time() < deadline - 5:
        angles_vfine = list(np.arange(best_angle - 0.5, best_angle + 0.5, 0.02))
        for ang in angles_vfine:
            if time.time() > deadline - 5:
                break
            side, cx, cy = eval_side_at_angle(ang)
            if side < best_side:
                best_side = side
                best_cx = cx
                best_cy = cy
                best_angle = ang
    
    return best_cx, best_cy, best_angle, best_side


def _optimize_outer_hex(inner_hexes, initial_cx, initial_cy, initial_angle, max_iters=80, tol=1e-10, deadline=None):
    """Optimize the outer hexagon (center, angle) to minimize its side length.
    Uses LP for center optimization at each angle, combined with multi-scale angle search."""
    if deadline is None:
        deadline = time.time() + 100
    
    best_cx, best_cy, best_angle = initial_cx, initial_cy, initial_angle
    sqrt3 = math.sqrt(3)
    
    # Precompute all vertices for faster evaluation
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    # Precompute direction vectors for the 6 normal directions
    dir_vectors = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        dir_vectors.append((math.cos(theta), math.sin(theta)))
    
    def eval_side_exact(cx, cy, angle):
        """Exact evaluation of outer hexagon side length."""
        a = -math.radians(angle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        max_val = 0.0
        for vx, vy in all_verts:
            dx = vx - cx
            dy = vy - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for nx, ny in dir_vectors:
                proj = lx * nx + ly * ny
                if proj > max_val:
                    max_val = proj
        return max_val * 2.0 / sqrt3
    
    def eval_side_lp(angle):
        """Use LP to find optimal center at given angle, return (side, cx, cy)."""
        a = -math.radians(angle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        
        c = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for vx, vy in all_verts:
            for nx, ny in dir_vectors:
                A_ub.append([nx, ny, -1.0])
                b_ub.append(-(vx * nx + vy * ny))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(-10, 10), (-10, 10), (0, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            opt_cx = result.x[0]
            opt_cy = result.x[1]
            t = result.x[2]
            side = t * 2.0 / sqrt3
            return side, opt_cx, opt_cy
        else:
            # Fallback
            max_val = 0.0
            for vx, vy in all_verts:
                dx = vx - best_cx
                dy = vy - best_cy
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                for nx, ny in dir_vectors:
                    proj = lx * nx + ly * ny
                    if proj > max_val:
                        max_val = proj
            side = max_val * 2.0 / sqrt3
            return side, best_cx, best_cy
    
    best_side = eval_side_exact(best_cx, best_cy, best_angle)
    
    # Phase 1: Coarse angle sweep with LP center
    step_a = 5.0
    for da in [step_a * i for i in range(-6, 7)]:
        if time.time() > deadline - 30:
            break
        new_angle = best_angle + da
        side, cx, cy = eval_side_lp(new_angle)
        if side < best_side - tol:
            best_angle = new_angle
            best_cx = cx
            best_cy = cy
            best_side = side
    
    # Phase 2: Medium angle sweep
    step_a = 1.0
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for da in [step_a * i for i in range(-10, 11)]:
            new_angle = best_angle + da
            side, cx, cy = eval_side_lp(new_angle)
            if side < best_side - tol:
                best_angle = new_angle
                best_cx = cx
                best_cy = cy
                best_side = side
                improved = True
    
    # Phase 3: Fine angle sweep
    step_a = 0.1
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for da in [step_a * i for i in range(-10, 11)]:
            new_angle = best_angle + da
            side, cx, cy = eval_side_lp(new_angle)
            if side < best_side - tol:
                best_angle = new_angle
                best_cx = cx
                best_cy = cy
                best_side = side
                improved = True
    
    # Phase 4: Very fine angle sweep
    step_a = 0.01
    for da in [step_a * i for i in range(-10, 11)]:
        if time.time() > deadline - 2:
            break
        new_angle = best_angle + da
        side, cx, cy = eval_side_lp(new_angle)
        if side < best_side - 1e-12:
            best_angle = new_angle
            best_cx = cx
            best_cy = cy
            best_side = side
    
    return best_cx, best_cy, best_angle, best_side


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 14 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse evaluation to find promising configurations
    for combo in all_combos:
        if time.time() > deadline - 40.0:  # Leave 40s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations with quick eval
        for oang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    # Phase 2: Fine-tune the best configuration with LP-based optimization
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Use LP-based optimization
        opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
            inner_hexes, best_center[0], best_center[1], best_angle,
            max_iters=80, deadline=deadline
        )
        
        if opt_side < best_side:
            best_side = opt_side
            best_center = (opt_cx, opt_cy)
            best_angle = opt_angle

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
            inner_hexes, cx_mean, cy_mean, 0.0, max_iters=80, deadline=deadline
        )
        best_center = (best_cx, best_cy)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
                    inner_hexes, cx_mean, cy_mean, 0.0, max_iters=80, deadline=deadline
                )
                best_center = (best_cx, best_cy)
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.001
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
