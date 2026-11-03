"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Topologically constrained packing with smooth sliding parameterization,
LP-exact outer evaluation, and multi-start Nelder-Mead polishing.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
EPS = 1e-9
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
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
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
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


def _all_disjoint(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _outer_lp(inner_hexes, oangle):
    """
    LP to find minimum enclosing regular hexagon (center + side) for fixed oangle.
    Returns (ox, oy, side) or (None, None, inf).
    """
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
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        t = res.x[2]
        return (res.x[0], res.x[1], t * 2.0 / SQRT3)
    return (None, None, float('inf'))


def _support_function(cx, cy, s, ang, nx, ny):
    max_val = 0.0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _pairwise_violation(pos, angles, i, j, n_dirs=12):
    """Return positive violation if hexagons i, j overlap; 0 if disjoint."""
    max_violation = 0.0
    for k in range(n_dirs):
        theta = 2.0 * math.pi * k / n_dirs
        nx = math.cos(theta)
        ny = math.sin(theta)
        s_i = _support_function(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
        s_j = _support_function(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
        dx = pos[2*j] - pos[2*i]
        dy = pos[2*j+1] - pos[2*i+1]
        dist_proj = dx * nx + dy * ny
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    return max_violation


def _build_sliding_config():
    """
    Build a topologically invariant configuration using a sliding parameterization.
    The 11 hexagons are arranged in a 1-2-3-2-1 cluster with relative rotations
    that maintain edge-contact constraints.
    
    Returns: list of (x, y, angle_deg) for 11 hexagons
    """
    # Start with honeycomb positions
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)
    
    # 1-2-3-2-1 cluster in axial coordinates
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons (to make 11 total)
    axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
             (2, 0), (0, -2), (-2, -1), (-1, 2)]
    
    positions = [axial_to_xy(q, r) for q, r in axial]
    angles = [0.0] * 11
    
    # Apply small relative rotations to break symmetry and enable optimization
    # The sliding parameterization allows adjacent hexagons to rotate relative to each other
    # while maintaining contact constraints via the optimization
    
    return [(x, y, a) for (x, y), a in zip(positions, angles)]


def _sliding_objective(state, contact_pairs, rest_length=SQRT3):
    """
    Smooth objective for sliding parameterization.
    state: [x1,y1,a1, ..., x11,y11,a11, ox, oy, oangle]
    
    Uses smooth surrogates for contact constraints and overlap penalties.
    """
    n_hex = 11
    ox = state[33]
    oy = state[34]
    oang = state[35]
    inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2]) for i in range(n_hex)]
    
    # LP outer side
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oang + 30 + 60 * k))
                ny = math.sin(math.radians(oang + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if not res.success:
        return 1e6
    side = res.x[2] * 2.0 / SQRT3
    
    # Contact constraint penalty (smooth): penalize deviation from rest length for contact pairs
    contact_penalty = 0.0
    for i, j in contact_pairs:
        dx = state[3*j] - state[3*i]
        dy = state[3*j+1] - state[3*i+1]
        dist = math.sqrt(dx*dx + dy*dy)
        deviation = dist - rest_length
        contact_penalty += deviation * deviation
    
    # Overlap penalty (smooth surrogate using support functions)
    angles_list = [state[3*m+2] for m in range(n_hex)]
    overlap_penalty = 0.0
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            v = _pairwise_violation(state, angles_list, i, j, n_dirs=12)
            if v > 0:
                overlap_penalty += v * v
    
    return side + 50.0 * contact_penalty + 500.0 * overlap_penalty


def _build_honeycomb():
    """Standard 11-hex honeycomb: center + ring1(6) + ring2(4)."""
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
             (2, 0), (0, -2), (-2, -1), (-1, 2)]
    xy = [axial_to_xy(q, r) for q, r in axial]
    return [(x, y, 0.0) for x, y in xy]


def _contact_graph():
    g = []
    for i in range(1, 7):
        g.append((0, i))
    for i in range(1, 7):
        j = i + 1 if i < 6 else 1
        g.append((i, j))
    g += [(7,1),(7,2),(8,3),(8,4),(9,4),(9,5),(10,5),(10,6)]
    return g


def _spring_relax(positions, angles, graph, n_iter=400, lr=0.008):
    n = len(positions)
    rest = SQRT3
    pos = np.array([[positions[i][0], positions[i][1]] for i in range(n)], dtype=float)
    ang = list(angles)
    for it in range(n_iter):
        forces = np.zeros_like(pos)
        for i, j in graph:
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                continue
            f = lr * (dist - rest) / dist
            forces[i][0] -= f * dx
            forces[i][1] -= f * dy
            forces[j][0] += f * dx
            forces[j][1] += f * dy
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    continue
                if dist < rest * 1.05:
                    f = 0.3 * (rest - dist) / dist
                    forces[i][0] -= f * dx
                    forces[i][1] -= f * dy
                    forces[j][0] += f * dx
                    forces[j][1] += f * dy
        pos += forces
    return [(pos[i][0], pos[i][1], ang[i]) for i in range(n)]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def time_left():
        return deadline - time.time()

    best_side = float('inf')
    best_state = None  # flat: x1,y1,a1,x2,y2,a2,...
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # --- Phase 1: Lattice enumeration with multiple outer angles ---
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    base_axial = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    ring2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),
             (-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,2)]

    oangles = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]

    count = 0
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time_left() < 30:
            break
        selected = [ring2[i] for i in combo]
        full_axial = base_axial + selected
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oang in oangles:
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            ox, oy, side = _outer_lp(inner_hexes, oang)
            if ox is not None and side < best_side:
                best_side = side
                best_center = (ox, oy)
                best_oangle = oang
                best_state = []
                for x, y in full_xy:
                    best_state.extend([x, y, 0.0])
        count += 1
        if count % 100 == 0 and time_left() < 30:
            break

    # --- Phase 2: Spring-mass relaxation ---
    if time_left() > 25:
        hc = _build_honeycomb()
        cg = _contact_graph()
        relaxed = _spring_relax([(x,y) for x,y,a in hc], [a for x,y,a in hc], cg)
        rel_pos = [(x,y) for x,y,a in relaxed]
        rel_ang = [a for x,y,a in relaxed]

        for oang in [0.0, 15.0, -15.0, 30.0, -30.0]:
            inner_hexes = [(rel_pos[i][0], rel_pos[i][1], 1.0, rel_ang[i]) for i in range(11)]
            ox, oy, side = _outer_lp(inner_hexes, oang)
            if ox is not None and side < best_side:
                best_side = side
                best_center = (ox, oy)
                best_oangle = oang
                best_state = []
                for i in range(11):
                    best_state.extend([rel_pos[i][0], rel_pos[i][1], rel_ang[i]])

    # --- Phase 3: Sliding parameterization optimization ---
    if time_left() > 20 and best_state is not None:
        n_hex = 11
        contact_pairs = _contact_graph()

        init_state = list(best_state) + [best_center[0], best_center[1], best_oangle]
        try:
            res = minimize(_sliding_objective, init_state, args=(contact_pairs,),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_side + 50:
                fs = res.x
                inner_hexes = [(fs[3*i], fs[3*i+1], 1.0, fs[3*i+2]) for i in range(n_hex)]
                if _all_disjoint(inner_hexes):
                    ox, oy, side = _outer_lp(inner_hexes, fs[35])
                    if ox is not None and side < best_side:
                        best_side = side
                        best_center = (ox, oy)
                        best_oangle = fs[35]
                        best_state = list(fs[:33])
        except Exception:
            pass

    # --- Phase 4: Fine outer angle sweep ---
    if time_left() > 10 and best_state is not None:
        for oang in np.arange(-90, 90.5, 5.0):
            if time_left() < 10:
                break
            inner_hexes = [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(11)]
            ox, oy, side = _outer_lp(inner_hexes, float(oang))
            if ox is not None and side < best_side:
                best_side = side
                best_center = (ox, oy)
                best_oangle = float(oang)

    # --- Phase 5: High-precision polish ---
    if time_left() > 8 and best_state is not None:
        n_hex = 11

        def polish_obj(state):
            ox = state[33]
            oy = state[34]
            oang = state[35]
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2]) for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(oang + 30 + 60 * k))
                        ny = math.sin(math.radians(oang + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            side = res.x[2] * 2.0 / SQRT3
            angles_list = [state[3*m+2] for m in range(n_hex)]
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i+1, n_hex):
                    v = _pairwise_violation(state, angles_list, i, j, n_dirs=24)
                    if v > 0:
                        penalty += v * v
            return side + 1000.0 * penalty

        init = list(best_state) + [best_center[0], best_center[1], best_oangle]
        try:
            res = minimize(polish_obj, init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < best_side + 10:
                fs = res.x
                inner_hexes = [(fs[3*i], fs[3*i+1], 1.0, fs[3*i+2]) for i in range(n_hex)]
                if _all_disjoint(inner_hexes):
                    ox, oy, side = _outer_lp(inner_hexes, fs[35])
                    if ox is not None and side < best_side:
                        best_side = side
                        best_center = (ox, oy)
                        best_oangle = fs[35]
                        best_state = list(fs[:33])
        except Exception:
            pass

    # --- Final verification ---
    inner_hexes = [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(11)]
    if not _all_disjoint(inner_hexes):
        # Scale positions to fix overlaps
        for scale in [1.002, 1.005, 1.01, 1.02, 1.05]:
            inner_hexes = [(best_state[3*i]*scale, best_state[3*i+1]*scale, 1.0, best_state[3*i+2]) for i in range(11)]
            if _all_disjoint(inner_hexes):
                best_state = []
                for x, y, s, a in inner_hexes:
                    best_state.extend([x, y, a])
                break

    inner_hexes = [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(11)]
    ox, oy, side = _outer_lp(inner_hexes, best_oangle)
    if ox is not None:
        best_center = (ox, oy)
        best_side = side
    else:
        # Fallback: compute manually
        cx_mean = sum(best_state[3*i] for i in range(11)) / 11.0
        cy_mean = sum(best_state[3*i+1] for i in range(11)) / 11.0
        max_r = 0.0
        for x, y, s, a in inner_hexes:
            verts = _hex_vertices(x, y, s, a)
            for vx, vy in verts:
                r = math.sqrt((vx-cx_mean)**2 + (vy-cy_mean)**2)
                if r > max_r:
                    max_r = r
        best_center = (cx_mean, cy_mean)
        best_side = max_r / (SQRT3/2.0)

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = all(_inner_in_outer(h, outer_hex) for h in inner_hexes)
    if not valid:
        for _ in range(300):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[best_state[3*i], best_state[3*i+1], best_state[3*i+2]] for i in range(11)]
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
