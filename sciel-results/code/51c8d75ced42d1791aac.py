"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses hexagonal lattice initial configuration + LP for outer hexagon + nonlinear polish.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE = 140.0  # seconds (150s limit with safety margin)
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60.0 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2.0
    for k in range(6):
        a_norm = math.radians(30.0 + 60.0 * k)
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
    dx = cx2 - cx1
    dy = cy2 - cy1
    # Quick reject: if centers are far apart, no overlap
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
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
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


def _min_enclosing_hex_lp(inner_hexes, oangle=0.0):
    """
    Find minimum side length of outer regular hexagon (fixed rotation oangle)
    that contains all inner hexagons. Uses LP to optimize center and scale.
    
    Variables: (cx, cy, t) where t = 2*apothem = side*sqrt(3)
    Minimize t
    Subject to: for each vertex (vx,vy) of each inner hex, and each face normal direction:
        nx*(vx-cx) + ny*(vy-cy) <= t
    """
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30.0 + 60.0 * k))
                ny = math.sin(math.radians(oangle + 30.0 + 60.0 * k))
                # nx*(vx-cx) + ny*(vy-cy) <= t  =>  -nx*cx - ny*cy - t <= -nx*vx - ny*vy
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1.0])
    b_ub.append(0.0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        side = t * 2.0 / SQRT3
        center = (res.x[0], res.x[1])
        return center, side
    else:
        return None, float('inf')


def _axial_to_xy(q, r):
    """Convert axial hex coordinates to Cartesian (for unit hexagons in lattice)."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _generate_lattice_config():
    """Generate the 11-hexagon lattice configuration: center + ring1 + 4 from ring2."""
    center = (0, 0)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    base = [center] + ring1  # 7 hexagons
    return base, ring2


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    base_axial, ring2 = _generate_lattice_config()
    
    # Phase 1: Try all combinations of 4 from ring2 with continuous outer angle optimization
    # Due to 3-fold symmetry, we can reduce the search space
    # The 14 ring-2 positions form a hexagonal ring. By 3-fold symmetry, 
    # choosing 4 positions has limited distinct configurations.
    
    # Generate all C(14,4) = 1001 combinations
    n_r2 = len(ring2)
    all_combos = list(itertools.combinations(range(n_r2), 4))
    
    # For each combination, try a range of outer angles
    # Key insight: the optimal outer angle depends on the symmetry of the inner config
    # Try angles from -30 to 30 in steps of 1 degree, then refine
    
    angles_coarse = list(range(-30, 31, 1))  # -30 to 30 step 1
    
    for combo in all_combos:
        if time.time() > deadline - 60.0:
            break
        
        selected_r2 = [ring2[i] for i in combo]
        full_axial = base_axial + selected_r2
        full_xy = [_axial_to_xy(q, r) for q, r in full_axial]
        
        for oangle in angles_coarse:
            if time.time() > deadline - 60.0:
                break
            
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _min_enclosing_hex_lp(inner_hexes, oangle)
            
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle
    
    if best_positions is None:
        # Fallback: just use center + ring1
        full_xy = [_axial_to_xy(q, r) for q, r in base_axial]
        best_positions = full_xy
        best_angles = [0.0] * 7
        center, side = _min_enclosing_hex_lp(
            [(x, y, 1.0, 0.0) for x, y in full_xy], 0.0)
        best_center = center
        best_side = side
        best_oangle = 0.0
    
    # Phase 2: Refine outer angle with finer grid around best
    if time.time() < deadline - 40.0:
        for oangle in np.arange(best_oangle - 3.0, best_oangle + 3.01, 0.1):
            if time.time() > deadline - 40.0:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) 
                          for i in range(len(best_positions))]
            center, side = _min_enclosing_hex_lp(inner_hexes, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
    # Phase 3: Nonlinear optimization to improve inner positions
    if time.time() < deadline - 20.0:
        n_hex = len(best_positions)
        
        def objective(state):
            """Objective: outer side length + penalty for overlaps."""
            angles = [state[3*m + 2] for m in range(n_hex)]
            hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2]) for i in range(n_hex)]
            
            # Compute outer hex side via LP
            center, side = _min_enclosing_hex_lp(hexes, best_oangle)
            if side == float('inf'):
                return 1e6
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    # Support function based separation
                    max_viol = 0.0
                    for k in range(12):
                        theta = math.radians(15.0 * k)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        
                        # Support of hex i in direction (nx, ny)
                        max_i = 0.0
                        for m in range(6):
                            a = math.radians(state[3*i+2] + 60.0 * m)
                            vx = math.cos(a)
                            vy = math.sin(a)
                            val = vx * nx + vy * ny
                            if val > max_i:
                                max_i = val
                        
                        # Support of hex j in direction (-nx, -ny)
                        max_j = 0.0
                        for m in range(6):
                            a = math.radians(state[3*j+2] + 60.0 * m)
                            vx = math.cos(a)
                            vy = math.sin(a)
                            val = vx * (-nx) + vy * (-ny)
                            if val > max_j:
                                max_j = val
                        
                        dx = state[3*j] - state[3*i]
                        dy = state[3*j+1] - state[3*i+1]
                        dist_proj = dx * nx + dy * ny
                        violation = max_i + max_j - dist_proj
                        if violation > max_viol:
                            max_viol = violation
                    
                    if max_viol > 0:
                        penalty += max_viol ** 2
            
            return side + 100.0 * penalty
        
        # Initial state
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        # Try a few perturbations
        np.random.seed(42)
        trial_states = [init_state[:]]
        for t in range(4):
            perturbed = init_state[:]
            for i in range(n_hex):
                perturbed[3*i] += np.random.uniform(-0.05, 0.05)
                perturbed[3*i+1] += np.random.uniform(-0.05, 0.05)
                perturbed[3*i+2] += np.random.uniform(-3.0, 3.0)
            trial_states.append(perturbed)
        
        for init_state_try in trial_states:
            if time.time() > deadline - 15.0:
                break
            try:
                result = minimize(objective, init_state_try, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                          'initial_simplex': 0.05})
                if result.fun < best_side + 10.0:
                    final_state = result.x
                    hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]) 
                             for i in range(n_hex)]
                    
                    if _all_disjoint(hexes):
                        center, side = _min_enclosing_hex_lp(hexes, best_oangle)
                        if side < best_side:
                            best_side = side
                            best_positions = [(h[0], h[1]) for h in hexes]
                            best_angles = [h[3] for h in hexes]
                            best_center = center
            except Exception:
                pass
    
    # Phase 4: Final polish - try to improve outer angle with very fine grid
    if time.time() < deadline - 10.0:
        for oangle in np.arange(best_oangle - 1.0, best_oangle + 1.01, 0.01):
            if time.time() > deadline - 10.0:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) 
                          for i in range(len(best_positions))]
            center, side = _min_enclosing_hex_lp(inner_hexes, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
    # Phase 5: Final verification and safety
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) 
                   for i in range(len(best_positions))]
    
    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            # Bump up side length slightly
            best_side *= 1.002
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            break
    
    # Ensure disjointness
    if not _all_disjoint(inner_hexes):
        # Try small nudges
        np.random.seed(123)
        for _ in range(20):
            new_hexes = []
            for h in inner_hexes:
                x, y, s, a = h
                new_hexes.append((x + np.random.uniform(-0.005, 0.005),
                                  y + np.random.uniform(-0.005, 0.005), s, a))
            if _all_disjoint(new_hexes):
                inner_hexes = new_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                center, side = _min_enclosing_hex_lp(inner_hexes, best_oangle)
                best_center = center
                best_side = side
                break
    
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] 
             for i in range(len(inner_hexes))]
    
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
