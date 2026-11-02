"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses convex optimization (LP for outer hexagon, nonlinear for inner positions).
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit


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


def _min_outer_side_lp(inner_hexes, oangle=0.0):
    """Find minimum side length of outer hexagon using LP."""
    # Variables: [cx, cy, t] where t is the apothem of outer hex
    # Objective: minimize t
    # For each inner vertex, for each of 6 directions of outer hex:
    #   (vx - cx) * nx_k + (vy - cy) * ny_k <= t
    c = [0, 0, 1]  # minimize t
    
    A_ub = []
    b_ub = []
    
    a_rad = math.radians(oangle)
    cos_a = math.cos(a_rad)
    sin_a = math.sin(a_rad)
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                # In outer hex local frame:
                # (vx - cx) * cos(theta) + (vy - cy) * sin(theta) <= t
                # => -vx*cos(theta) + cx*cos(theta) - vy*sin(theta) + cy*sin(theta) <= t
                # => cx*cos(theta) + cy*sin(theta) - t <= vx*cos(theta) + vy*sin(theta)
                # But we need to account for rotation of outer hex...
                # Actually the outer hex is rotated by oangle, so its normals are at angles oangle + 30 + 60k
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                # (vx - cx)*nx + (vy - cy)*ny <= t
                # -nx*cx - ny*cy - t <= -vx*nx - vy*ny
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    # Also need t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        # side = 2 * apothem / sqrt(3)
        return t * 2.0 / math.sqrt(3)
    else:
        return float('inf')


def _compute_outer_for_positions(positions, angles, oangle=0.0):
    """Compute optimal outer hexagon for given inner positions and angles."""
    inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(len(positions))]
    side = _min_outer_side_lp(inner_hexes, oangle)
    # Get center from LP
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        center = (res.x[0], res.x[1])
    else:
        # Fallback to centroid
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    
    return center, side


def _separation_distance(hex1, hex2, direction):
    """Minimum distance along direction needed for two hexagons to be disjoint."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    # Support function of hexagon in direction (nx, ny):
    # max over vertices of (v - c) . (nx, ny)
    def support(cx, cy, s, ang, nx, ny):
        max_val = 0
        for k in range(6):
            a = math.radians(ang + 60 * k)
            vx = s * math.cos(a)
            vy = s * math.sin(a)
            val = vx * nx + vy * ny
            if val > max_val:
                max_val = val
        return max_val
    
    # For disjointness in direction d: (c2 - c1) . d >= support1(d) + support2(-d)
    # = support1(d) + support2(d) [by symmetry of hexagon]
    s1_dir = support(cx1, cy1, s1, a1, direction[0], direction[1])
    s2_dir = support(cx2, cy2, s2, a2, -direction[0], -direction[1])
    return s1_dir + s2_dir


def _pairwise_separation_constraint(pos, angles, i, j, oangle=0.0):
    """Check if hexagons i and j are disjoint. Return violation (negative = OK)."""
    hex_i = (pos[2*i], pos[2*i+1], 1.0, angles[i])
    hex_j = (pos[2*j], pos[2*j+1], 1.0, angles[j])
    
    # Check in multiple directions
    max_violation = 0
    for k in range(12):
        theta = math.radians(15 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        def support(cx, cy, s, ang, nx, ny):
            max_val = 0
            for m in range(6):
                a = math.radians(ang + 60 * m)
                vx = s * math.cos(a)
                vy = s * math.sin(a)
                val = vx * nx + vy * ny
                if val > max_val:
                    max_val = val
            return max_val
        
        s_i = support(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
        s_j = support(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
        
        dx = pos[2*j] - pos[2*i]
        dy = pos[2*j+1] - pos[2*i+1]
        dist_proj = dx * nx + dy * ny
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    
    return max_violation


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Generate lattice positions: center + ring1 + ring2
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)  # 7 positions

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration with multiple outer orientations
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0]
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    # Phase 2: Nonlinear optimization of inner positions and angles
    if time.time() < deadline - 5:
        n_hex = len(best_positions)

        def objective(state):
            # Compute outer side via LP (exact)
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / sqrt3

            # Penalty for overlaps
            penalty = 0.0
            angles_list = [state[3*m+2] for m in range(n_hex)]
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation_constraint(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side + 200.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_side + 50:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 3: Optimize outer angle with current best positions
    if time.time() < deadline - 10:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                           for i in range(len(best_positions))]
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 4: Differential evolution for global search
    if time.time() < deadline - 15:
        n_hex = len(best_positions)
        
        def de_objective(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / sqrt3

            penalty = 0.0
            angles_list = [state[3*m+2] for m in range(n_hex)]
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation_constraint(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side + 500.0 * penalty

        # Bounds for differential evolution
        bounds = []
        for i in range(n_hex):
            bounds.append((-5, 5))  # x
            bounds.append((-5, 5))  # y
            bounds.append((-30, 30))  # angle

        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=30,
                                                popsize=15, tol=1e-8, seed=42,
                                                polish=False, init='sobol')
            if de_result.fun < best_side + 10:
                final_state = de_result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 5: High-precision polish - local refinement around best solution
    if time.time() < deadline - 5:
        n_hex = len(best_positions)

        def polish_objective(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / sqrt3
            penalty = 0.0
            angles_list = [state[3*m+2] for m in range(n_hex)]
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation_constraint(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side + 500.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(polish_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 10:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(200):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
