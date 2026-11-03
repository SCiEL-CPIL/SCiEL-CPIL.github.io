"""Combined program: LP-based outer hexagon with continuous angle search + inner polish."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length) or None on failure."""
    # Build vertex array once
    verts_list = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        verts_list.extend(verts)
    
    n_pts = len(verts_list)
    if n_pts == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy arrays for speed
    vx_arr = np.array([v[0] for v in verts_list])
    vy_arr = np.array([v[1] for v in verts_list])
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = np.array([0.0, 0.0, 1.0])
    
    # Build A_ub and b_ub using numpy for speed
    A_rows = []
    b_vals = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        A_rows.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
        b_vals.append(-nx * vx_arr - ny * vy_arr)
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        A_rows.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
        b_vals.append(nx * vx_arr + ny * vy_arr)
    
    # t >= 0
    A_rows.append(np.array([[0.0, 0.0, -1.0]]))
    b_vals.append(np.array([0.0]))
    
    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_vals)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                        bounds=[(None, None), (None, None), (0, None)], 
                        method='highs')
        if result.success:
            cx, cy, t = result.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except:
        pass
    
    return None

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Uses continuous angle search. Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search: 0 to 30 degrees (due to 60-degree symmetry)
    # Use step of 1.0 to reduce LP calls
    for angle_deg in np.arange(0, 30.01, 1.0):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = angle_deg
    
    # Fine search around best coarse angle
    if best_side < float('inf'):
        # First pass: step 0.1
        search_range = 1.0
        step = 0.1
        for angle_deg in np.arange(max(0, best_angle - search_range), 
                                   min(30, best_angle + search_range) + step, step):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
        
        # Second pass: step 0.01 around the best
        search_range = 0.1
        step = 0.01
        for angle_deg in np.arange(max(0, best_angle - search_range), 
                                   min(30, best_angle + search_range) + step, step):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def _hexagons_overlap(x1, y1, a1, x2, y2, a2, eps=0.001):
    """Check if two unit hexagons overlap using support function distance."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Convert to radians
    a1_rad = math.radians(a1)
    a2_rad = math.radians(a2)
    
    # For two regular hexagons of side 1, the minimum distance between centers
    # for non-overlap depends on their relative orientation
    # The support function of a unit hexagon in direction theta is:
    # h(theta) = cos(theta - alpha) / cos(pi/6) for the appropriate face
    # where alpha is the orientation of the hexagon
    
    # The distance between centers must be at least h1(theta) + h2(theta + pi)
    # for all directions theta, where theta is the direction from hex1 to hex2
    
    if dist < 1e-10:
        return True  # Same center, definitely overlap
    
    theta = math.atan2(dy, dx)
    
    # For hexagon 1, compute support in direction theta
    # A regular hexagon with orientation alpha has vertices at alpha + k*pi/3
    # The face normals are at alpha + pi/6 + k*pi/3
    # Support function h(theta) = max over vertices of dot(vertex, dir)
    
    # Compute support of hex1 in direction theta
    h1 = 0.0
    for k in range(6):
        v_angle = a1_rad + math.pi * k / 3
        vx = math.cos(v_angle)
        vy = math.sin(v_angle)
        proj = vx * math.cos(theta) + vy * math.sin(theta)
        if proj > h1:
            h1 = proj
    
    # Compute support of hex2 in direction theta + pi (opposite direction)
    h2 = 0.0
    for k in range(6):
        v_angle = a2_rad + math.pi * k / 3
        vx = math.cos(v_angle)
        vy = math.sin(v_angle)
        proj = vx * math.cos(theta + math.pi) + vy * math.sin(theta + math.pi)
        if proj > h2:
            h2 = proj
    
    # The hexagons overlap if dist < h1 + h2 - eps
    return dist < (h1 + h2 - eps)

def _check_disjoint(inner_hexes):
    """Check if all inner hexagons have disjoint interiors."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            if _hexagons_overlap(x1, y1, a1, x2, y2, a2):
                return False
    return True

def _nelder_mead_polish(inner_hexes, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side
    with smooth signed-distance-based penalty for collisions."""
    
    def _smooth_collision_penalty(params):
        """Compute smooth collision penalty using signed distance between hexagons."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                x1, y1, a1 = params[i*3], params[i*3+1], params[i*3+2]
                x2, y2, a2 = params[j*3], params[j*3+1], params[j*3+2]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-10:
                    penalty += 10.0
                    continue
                
                theta = math.atan2(dy, dx)
                
                # Compute support of hex1 in direction theta
                a1_rad = math.radians(a1)
                h1 = 0.0
                for k in range(6):
                    v_angle = a1_rad + math.pi * k / 3
                    vx = math.cos(v_angle)
                    vy = math.sin(v_angle)
                    proj = vx * math.cos(theta) + vy * math.sin(theta)
                    if proj > h1:
                        h1 = proj
                
                # Compute support of hex2 in direction theta + pi
                a2_rad = math.radians(a2)
                h2 = 0.0
                for k in range(6):
                    v_angle = a2_rad + math.pi * k / 3
                    vx = math.cos(v_angle)
                    vy = math.sin(v_angle)
                    proj = vx * math.cos(theta + math.pi) + vy * math.sin(theta + math.pi)
                    if proj > h2:
                        h2 = proj
                
                # Signed distance: positive if separated, negative if overlapping
                sdist = dist - (h1 + h2)
                if sdist < 0:
                    # Smooth penalty: quadratic in penetration depth
                    penalty += (-sdist) ** 2 * 100
        
        return penalty
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Smooth collision penalty
        col_penalty = _smooth_collision_penalty(params)
        
        # If significant collision, return large value
        if col_penalty > 1.0:
            return 1000.0 + col_penalty
        
        cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
        return side + col_penalty * 0.1
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        
        if result.fun < 1000.0:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            
            # Verify the polished solution is valid (no overlaps)
            if _check_disjoint(polished):
                return polished
    except:
        pass
    
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points, ensuring no overlaps
    positions = []
    for _, x, y in candidates:
        # Check this position doesn't overlap with existing positions
        valid = True
        for px, py in positions:
            if _hexagons_overlap(x, y, 0.0, px, py, 0.0, eps=0.002):
                valid = False
                break
        if valid:
            positions.append((x, y))
        if len(positions) == N:
            break
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        
        # Compute initial minimum enclosing hexagon
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_inner = [h[:] for h in inner]
            best_center = (cx, cy)
            best_outer_angle = angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try Nelder-Mead polish if time permits
    if time.time() < _DEADLINE - 30.0:
        polished = _nelder_mead_polish(best_inner, max_iter=150, tol=1e-5)
        if polished is not best_inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6 and _check_disjoint(polished):
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_outer_angle = pangle
    
    # Second round of polish with tighter tolerance if time permits
    if time.time() < _DEADLINE - 15.0:
        polished2 = _nelder_mead_polish(best_inner, max_iter=80, tol=1e-7)
        if polished2 is not best_inner:
            pcx2, pcy2, pside2, pangle2 = _min_enclosing_hex_side(polished2)
            if pside2 < best_side - 1e-7 and _check_disjoint(polished2):
                best_side = pside2
                best_inner = polished2
                best_center = (pcx2, pcy2)
                best_outer_angle = pangle2
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 10.0:
        # Try a few more angles around the best angle with finer steps
        for da in np.arange(-1.0, 1.01, 0.05):
            test_angle = (best_outer_angle + da) % 60
            if test_angle < 0 or test_angle > 30:
                continue
            res = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if res:
                tcx, tcy, tside = res
                if tside < best_side - 1e-10:
                    best_side = tside
                    best_center = (tcx, tcy)
                    best_outer_angle = test_angle
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_outer_angle
