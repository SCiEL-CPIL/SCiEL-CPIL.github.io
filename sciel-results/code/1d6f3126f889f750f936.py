"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    The outer hexagon with center (cx,cy), inradius r, and rotation angle_deg has
    6 half-planes. A point (vx,vy) is inside iff for all k:
      (vx-cx)*nx_k + (vy-cy)*ny_k <= r
    where (nx_k, ny_k) = (cos(theta_k), sin(theta_k)), theta_k = angle_deg + 30 + 60k degrees.
    
    We want to minimize r subject to these constraints for all vertices.
    This is a linear program in variables (cx, cy, r).
    """
    # Collect all vertices
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # LP: minimize r
    # Variables: [cx, cy, r]
    # Constraints: for each vertex (vx, vy) and each direction k:
    #   (vx - cx) * nx_k + (vy - cy) * ny_k <= r
    #   => -nx_k * cx - ny_k * cy - r <= -vx*nx_k - vy*ny_k
    # Also: r >= 0
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    # r >= 0  =>  -r <= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        # Fallback: use centroid and compute required r
        cx_mean = sum(v[0] for v in all_vertices) / n_pts
        cy_mean = sum(v[1] for v in all_vertices) / n_pts
        r_opt = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx = math.cos(a)
            ny = math.sin(a)
            for vx, vy in all_vertices:
                proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                r_opt = max(r_opt, proj)
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute all vertices once
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # First pass: coarse sweep
    angles_coarse = np.arange(0, 60, 1.0)  # 0, 1, 2, ..., 59
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Second pass: fine sweep around the best angle
    if best_side < float('inf'):
        angles_fine = np.arange(best_angle - 3, best_angle + 3, 0.1)
        for angle_deg in angles_fine:
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(best_angle - 0.5, best_angle + 0.5, 0.01)
        for angle_deg in angles_finer:
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Fourth pass: Nelder-Mead polish on (cx, cy, angle) using LP as oracle
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        from scipy.optimize import minimize
        
        def objective(params):
            cx, cy, angle_deg = params
            # Wrap angle to [0, 60)
            angle_deg = angle_deg % 60.0
            # Compute required inradius r for this center and angle
            r_max = 0.0
            for k in range(6):
                a = math.radians(angle_deg + 30 + 60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                for vx, vy in all_vertices:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > r_max:
                        r_max = proj
            # Convert to side length
            return r_max / (math.sqrt(3) / 2)
        
        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success:
                cx_opt, cy_opt, ang_opt = res.x
                ang_opt = ang_opt % 60.0
                side_opt = res.fun
                if side_opt < best_side:
                    best_side = side_opt
                    best_center = (cx_opt, cy_opt)
                    best_angle = ang_opt
        except Exception:
            pass
        
        # Fifth pass: very fine angle sweep around polished angle
        if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
            angles_ultra = np.arange(best_angle - 0.1, best_angle + 0.1, 0.001)
            for angle_deg in angles_ultra:
                if angle_deg < 0 or angle_deg >= 60:
                    continue
                cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 30 (Pointy-Top)
    # This aligns the lattice vectors (0 and 60 deg) with the flat-to-flat normals,
    # requiring a minimum separation of sqrt(3) instead of 2.
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # Try alternative arrangements: all angles = 0 (Flat-Top)
    inner_alt = [[pos[0], pos[1], 0.0] for pos in positions]
    
    # Also try a denser packing: reduce scale slightly to allow tighter outer hex
    # but keep strict disjointness
    scale2 = 1.0001
    u2 = (sqrt3 * scale2, 0.0)
    v2 = (sqrt3 / 2 * scale2, 1.5 * scale2)
    candidates2 = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u2[0] + j * v2[0]
            y = i * u2[1] + j * v2[1]
            dist_sq = x * x + y * y
            candidates2.append((dist_sq, x, y))
    candidates2.sort(key=lambda t: t[0])
    positions2 = []
    for _, x, y in candidates2:
        positions2.append((x, y))
        if len(positions2) == N:
            break
    inner_alt2 = [[pos[0], pos[1], 30.0] for pos in positions2]
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        # Add circumradius 1 to max_dist for a rough bound
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    # Try the flat-top arrangement
    if time.time() < _DEADLINE - 20.0:
        cx2, cy2, side2, angle2 = _min_enclosing_hex_side(inner_alt)
        if side2 < side:
            inner, cx, cy, side, angle = inner_alt, cx2, cy2, side2, angle2
    
    # Try the denser packing
    if time.time() < _DEADLINE - 20.0:
        cx3, cy3, side3, angle3 = _min_enclosing_hex_side(inner_alt2)
        if side3 < side:
            inner, cx, cy, side, angle = inner_alt2, cx3, cy3, side3, angle3
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
