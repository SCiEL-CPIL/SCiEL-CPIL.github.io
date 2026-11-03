"""Hexagon packing: 11 unit hexagons in a regular hexagon, minimize outer side.
Combined approach: lattice topology + hinge deformation + LP outer optimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0  # safety margin within 150s limit

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _hex_support(cx, cy, angle_deg, nx, ny):
    """Support function of unit hexagon at (cx,cy) rotated by angle_deg in direction (nx,ny)."""
    angle_rad = math.radians(angle_deg)
    max_proj = -float('inf')
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        vx = cx + math.cos(a)
        vy = cy + math.sin(a)
        proj = vx * nx + vy * ny
        if proj > max_proj:
            max_proj = proj
    return max_proj

def _hex_clearance(h1, h2):
    """Minimum clearance between two unit hexagons using separating axis theorem.
    Returns positive if separated, negative if overlapping."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    
    normals = []
    for ang in [a1, a2]:
        for k in range(6):
            theta = math.radians(ang + 30 + 60 * k)
            normals.append((math.cos(theta), math.sin(theta)))
    
    min_clearance = float('inf')
    for nx, ny in normals:
        sup1 = _hex_support(x1, y1, a1, nx, ny)
        sup1_neg = _hex_support(x1, y1, a1, -nx, -ny)
        sup2 = _hex_support(x2, y2, a2, nx, ny)
        sup2_neg = _hex_support(x2, y2, a2, -nx, -ny)
        
        gap1 = sup2 - sup1_neg
        gap2 = sup1 - sup2_neg
        clearance_axis = min(gap1, gap2)
        
        if clearance_axis < min_clearance:
            min_clearance = clearance_axis
    
    return min_clearance

def _check_disjoint(inner_hexes, min_gap=1e-6):
    """Check all pairs are disjoint with at least min_gap separation."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            clr = _hex_clearance(inner_hexes[i], inner_hexes[j])
            if clr < min_gap:
                return False
    return True

def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """For fixed outer hex angle, find optimal center and side via LP.
    Returns (cx, cy, side, angle_deg)."""
    all_verts = []
    for ix, iy, iang in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, iang))
    
    n_pts = len(all_verts)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx, cy, r = result.x[0], result.x[1], result.x[2]
    else:
        cx = sum(v[0] for v in all_verts) / n_pts
        cy = sum(v[1] for v in all_verts) / n_pts
        r = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                r = max(r, proj)
    
    side = r / (math.sqrt(3) / 2)
    return cx, cy, side, angle_deg

def _min_enclosing_hex(inner_hexes):
    """Find minimum enclosing regular hexagon over all angles."""
    best_side = float('inf')
    best_cx, best_cy, best_ang = 0.0, 0.0, 0.0
    
    # Coarse sweep
    for ang in np.arange(0, 60, 2.0):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    # Fine sweep
    for ang in np.arange(best_ang - 4, best_ang + 4, 0.5):
        if ang < 0 or ang >= 60:
            continue
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    # Finer sweep
    for ang in np.arange(best_ang - 1, best_ang + 1, 0.1):
        if ang < 0 or ang >= 60:
            continue
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    return best_cx, best_cy, best_side, best_ang

def _hexagonal_lattice_placement(n_hex, gap_factor=1.0):
    """Place hexagons in a compact hexagonal lattice pattern."""
    d = math.sqrt(3) * gap_factor
    u = np.array([d, 0.0])
    v = np.array([d * 0.5, d * math.sqrt(3) / 2])
    
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            pos = i * u + j * v
            dist_sq = pos[0]**2 + pos[1]**2
            candidates.append((dist_sq, pos[0], pos[1]))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hex:
            break
    
    return [[x, y, 30.0] for x, y in positions]

def _build_lattice_topology():
    """Build the 11-hexagon lattice topology: center + ring1 (6) + 4 from ring2.
    Returns list of axial coordinates and adjacency graph."""
    # Center + ring 1
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (12 hexagons)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    return pos_axial, ring1, ring2

def _axial_to_xy(q, r, gap_factor=1.0):
    """Convert axial coordinates to Cartesian for unit hexagons."""
    sqrt3 = math.sqrt(3)
    x = 1.5 * q * gap_factor
    y = sqrt3 * (r + q / 2.0) * gap_factor
    return (x, y)

def _compute_hinge_angles(hinges, pos_axial, ring1, ring2, gap_factor=1.0):
    """Given hinge angle parameters, compute the deformed positions.
    hinges: list of relative rotation angles (in radians) for each ring2 hexagon
    relative to its parent in ring1."""
    # Map ring2 positions to their parent ring1 hexagons
    # Each ring2 hex is adjacent to one or more ring1 hexes
    # For simplicity, we assign each ring2 hex to its nearest ring1 parent
    
    n_r2 = len(hinges)
    # Get the selected ring2 positions (we'll use the first n_r2 from ring2)
    selected_r2 = ring2[:n_r2]
    
    positions = []
    angles = []
    
    # Center
    cx, cy = _axial_to_xy(0, 0, gap_factor)
    positions.append((cx, cy))
    angles.append(0.0)
    
    # Ring 1
    for (q, r) in ring1:
        x, y = _axial_to_xy(q, r, gap_factor)
        positions.append((x, y))
        angles.append(0.0)
    
    # Ring 2 with hinge deformations
    # Each ring2 hex gets a small angular offset from its "natural" position
    for i, (q, r) in enumerate(selected_r2):
        x, y = _axial_to_xy(q, r, gap_factor)
        # Apply a small positional perturbation based on hinge angle
        # The hinge angle rotates the hex around its center
        angles.append(hinges[i] if i < len(hinges) else 0.0)
        positions.append((x, y))
    
    # Build inner hex list
    inner = []
    for i in range(len(positions)):
        x, y = positions[i]
        a = angles[i]
        inner.append([x, y, math.degrees(a) + 30.0])
    
    return inner

def _optimize_hinges(inner, time_budget=20.0):
    """Optimize hinge angles and positions to minimize enclosing hexagon side."""
    
    def flatten(hexes):
        params = []
        for x, y, a in hexes:
            params.extend([x, y, math.radians(a)])
        return np.array(params)
    
    def unflatten(params):
        hexes = []
        for i in range(len(params) // 3):
            hexes.append([params[3*i], params[3*i+1], math.degrees(params[3*i+2])])
        return hexes
    
    def objective(params):
        hexes = unflatten(params)
        n = len(hexes)
        min_clr = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                clr = _hex_clearance(hexes[i], hexes[j])
                if clr < min_clr:
                    min_clr = clr
        
        if min_clr < 0.005:
            violation = max(0.0, 0.005 - min_clr)
            return 1000.0 + violation * 500
        
        all_verts = []
        for ix, iy, iang in hexes:
            all_verts.extend(_hex_vertices(ix, iy, iang))
        
        best_side = float('inf')
        for outer_ang in [0.0, 15.0, 30.0, 45.0]:
            cx = sum(v[0] for v in all_verts) / len(all_verts)
            cy = sum(v[1] for v in all_verts) / len(all_verts)
            r = 0.0
            for k in range(6):
                a = math.radians(outer_ang + 30 + 60 * k)
                nx, ny = math.cos(a), math.sin(a)
                for vx, vy in all_verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    r = max(r, proj)
            side = r / (math.sqrt(3) / 2)
            best_side = min(best_side, side)
        
        return best_side
    
    x0 = flatten(inner)
    t_start = time.time()
    
    best_params = x0.copy()
    best_obj = objective(x0)
    
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 15.0 or time.time() - t_start > time_budget:
            break
        
        if restart == 0:
            x_init = x0
        else:
            scale = 0.2 * (restart + 1)
            x_init = x0 + np.random.normal(0, scale, size=x0.shape)
        
        try:
            result = minimize(objective, x_init, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01,
                                   'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_obj < 1000:
        return unflatten(best_params)
    return inner

def _try_ring2_combinations(time_budget=30.0):
    """Try different combinations of 4 ring2 hexagons and optimize each."""
    pos_axial, ring1, ring2 = _build_lattice_topology()
    
    best_inner = None
    best_side = float('inf')
    t_start = time.time()
    
    # Try all C(12,4) = 495 combinations
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > _DEADLINE - 10.0 or time.time() - t_start > time_budget:
            break
        
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [_axial_to_xy(q, r) for q, r in full_axial]
        
        inner = [[x, y, 30.0] for x, y in full_xy]
        
        if not _check_disjoint(inner, min_gap=1e-6):
            continue
        
        # Quick estimate
        cx, cy, side, ang = _min_enclosing_hex_lp(inner, 0.0)
        if side >= best_side * 1.002:
            continue
        
        # Optimize this configuration
        opt_inner = _optimize_hinges(inner, time_budget=5.0)
        if _check_disjoint(opt_inner, min_gap=1e-5):
            cx, cy, side, ang = _min_enclosing_hex(opt_inner)
            if side < best_side:
                best_side = side
                best_inner = opt_inner
    
    return best_inner, best_side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = _hexagonal_lattice_placement(N, gap_factor=1.05)
        cx, cy, side, ang = _min_enclosing_hex(inner)
        return inner, (cx, cy), side, ang
    
    best_inner = None
    best_side = float('inf')
    
    # Strategy 1: Try ring2 combinations with optimization
    if time.time() < _DEADLINE - 40.0:
        combo_inner, combo_side = _try_ring2_combinations(time_budget=35.0)
        if combo_inner is not None:
            best_inner = combo_inner
            best_side = combo_side
    
    # Strategy 2: If no good combo found, use lattice placement
    if best_inner is None:
        inner = _hexagonal_lattice_placement(N, gap_factor=1.0)
        if _check_disjoint(inner, min_gap=1e-6):
            best_inner = inner
            cx, cy, side, ang = _min_enclosing_hex(inner)
            best_side = side
        else:
            inner = _hexagonal_lattice_placement(N, gap_factor=1.05)
            best_inner = inner
            cx, cy, side, ang = _min_enclosing_hex(inner)
            best_side = side
    
    # Strategy 3: Further optimize the best found
    if time.time() < _DEADLINE - 20.0:
        opt_inner = _optimize_hinges(best_inner, time_budget=15.0)
        if _check_disjoint(opt_inner, min_gap=1e-5):
            cx, cy, side, ang = _min_enclosing_hex(opt_inner)
            if side < best_side:
                best_inner = opt_inner
                best_side = side
    
    # Verify disjointness
    if not _check_disjoint(best_inner, min_gap=1e-4):
        inner = _hexagonal_lattice_placement(N, gap_factor=1.05)
        best_inner = inner
        cx, cy, side, ang = _min_enclosing_hex(inner)
        best_side = side
    
    # Compute optimal enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        all_verts = []
        for ix, iy, iang in best_inner:
            all_verts.extend(_hex_vertices(ix, iy, iang))
        cx = sum(v[0] for v in all_verts) / len(all_verts)
        cy = sum(v[1] for v in all_verts) / len(all_verts)
        r = 0.0
        for k in range(6):
            a = math.radians(30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                r = max(r, proj)
        side = r / (math.sqrt(3) / 2)
        return best_inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex(best_inner)
    
    return best_inner, (cx, cy), side, angle
