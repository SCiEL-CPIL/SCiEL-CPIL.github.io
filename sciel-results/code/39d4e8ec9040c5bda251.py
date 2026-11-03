"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

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
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length)."""
    from scipy.optimize import linprog
    
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    n = 6  # 6 constraints per direction (max and min)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy + nx*vx + ny*vy <= t
        # => -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - nx*vx - ny*vy <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if result.success:
        cx, cy, t = result.x
        side = t / (math.sqrt(3) / 2)
        return cx, cy, side
    
    # Fallback: use centroid
    cx = sum(v[0] for v in all_vertices) / n_pts
    cy = sum(v[1] for v in all_vertices) / n_pts
    required_inradius = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        max_proj = -float('inf')
        min_proj = float('inf')
        for vx, vy in all_vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            max_proj = max(max_proj, proj)
            min_proj = min(min_proj, proj)
        required_inradius = max(required_inradius, max(max_proj, -min_proj))
    side = required_inradius / (math.sqrt(3) / 2)
    return cx, cy, side

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different angles for the outer hexagon (0 to 60 degrees, step 5)
    for angle_deg in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
        cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle_deg
    
    # Refine around best angle
    for da in [-3, -2, -1, 0, 1, 2, 3]:
        test_angle = (best_angle + da) % 60
        cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, test_angle)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = test_angle
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes, eps=1e-9):
    """Check if all inner hexagons have disjoint interiors using SAT test."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _hexagons_overlap(h1, h2, eps=1e-9):
    """Check if two hexagons overlap using SAT (Separating Axis Theorem)."""
    verts1 = _get_hex_vertices(h1[0], h1[1], h1[2])
    verts2 = _get_hex_vertices(h2[0], h2[1], h2[2])
    
    # Collect all edge normals from both hexagons
    axes = []
    for verts in [verts1, verts2]:
        for k in range(6):
            x1, y1 = verts[k]
            x2, y2 = verts[(k + 1) % 6]
            # Edge vector
            dx = x2 - x1
            dy = y2 - y1
            # Normal (perpendicular)
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            nx = -dy / length
            ny = dx / length
            axes.append((nx, ny))
    
    # Test each axis
    for nx, ny in axes:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        
        for vx, vy in verts1:
            proj = vx * nx + vy * ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        
        for vx, vy in verts2:
            proj = vx * nx + vy * ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        
        # If there's a gap, no overlap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def _nelder_mead_polish(inner_hexes, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using L-BFGS-B with soft-max surrogate."""
    from scipy.optimize import minimize
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness using proper hexagon overlap test
        for i in range(N):
            for j in range(i + 1, N):
                if _hexagons_overlap(new_inner[i], new_inner[j], eps=-1e-4):
                    # Compute penetration depth approximation
                    dx = new_inner[i][0] - new_inner[j][0]
                    dy = new_inner[i][1] - new_inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Penalty based on how much they overlap
                    min_dist = math.sqrt(3)
                    if dist < min_dist:
                        return 1000.0 + (min_dist - dist) * 50
                    return 1000.0 + 10
        
        cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
        return side
    
    def soft_objective(params, beta=50.0):
        """Smooth surrogate using log-sum-exp for the max in support function."""
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                if _hexagons_overlap(new_inner[i], new_inner[j], eps=-1e-4):
                    dx = new_inner[i][0] - new_inner[j][0]
                    dy = new_inner[i][1] - new_inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = math.sqrt(3)
                    if dist < min_dist:
                        return 1000.0 + (min_dist - dist) * 50
                    return 1000.0 + 10
        
        # Compute all vertices
        all_vertices = []
        for ix, iy, iangle in new_inner:
            verts = _get_hex_vertices(ix, iy, iangle)
            all_vertices.extend(verts)
        
        n_pts = len(all_vertices)
        best_t = 0.0
        
        # Try a few angles for outer hexagon
        for angle_deg in [0.0, 30.0]:
            # Compute required inradius t for this angle
            # t = max over all vertices and all 6 directions of |proj - center_proj|
            # But we also need to optimize center, so use LP-like approach
            # For speed, use centroid as initial center estimate
            cx = sum(v[0] for v in all_vertices) / n_pts
            cy = sum(v[1] for v in all_vertices) / n_pts
            
            t_val = 0.0
            for k in range(6):
                a = math.radians(angle_deg + 30 + 60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                
                max_proj = float('-inf')
                min_proj = float('inf')
                for vx, vy in all_vertices:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    max_proj = max(max_proj, proj)
                    min_proj = min(min_proj, proj)
                
                t_val = max(t_val, max_proj, -min_proj)
            
            best_t = max(best_t, t_val)
        
        side = best_t / (math.sqrt(3) / 2)
        return side
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    # Use L-BFGS-B with bounds
    bounds = [(-10, 10), (-10, 10), (-90, 90)] * N
    
    try:
        result = minimize(soft_objective, x0, method='L-BFGS-B', 
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': max_iter * 10})
        
        if result.success and result.fun < 1000.0:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            
            # Verify with exact objective
            exact_val = objective(result.x)
            if exact_val < 1000.0:
                return polished
    except Exception:
        pass
    
    # Fallback to Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        
        if result.success and result.fun < 1000.0:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            return polished
    except Exception:
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
    
    # Compute initial minimum enclosing hexagon
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
    best_side = side
    best_inner = [h[:] for h in inner]
    best_center = (cx, cy)
    best_angle = angle
    
    # Try Nelder-Mead polish if time permits
    if time.time() < _DEADLINE - 30.0:
        polished = _nelder_mead_polish(inner, max_iter=100, tol=1e-5)
        if polished is not inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 15.0:
        # Try a fine grid of angles around the best angle
        for da in range(-10, 11, 2):
            test_angle = (best_angle + da) % 60
            tcx, tcy, tside = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if tside < best_side - 1e-8:
                best_side = tside
                best_center = (tcx, tcy)
                best_angle = test_angle
        
        # Even finer refinement
        for da in [x * 0.5 for x in range(-10, 11, 1)]:
            test_angle = (best_angle + da) % 60
            tcx, tcy, tside = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if tside < best_side - 1e-9:
                best_side = tside
                best_center = (tcx, tcy)
                best_angle = test_angle
    
    # Verify the solution is valid
    if not _check_disjoint(best_inner, eps=-1e-6):
        # Fallback to a known valid configuration
        sqrt3 = math.sqrt(3)
        scale = 1.001
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        best_inner = [[pos[0], pos[1], 30.0] for pos in positions]
        cx, cy, side, angle = _min_enclosing_hex_side(best_inner)
        best_side = side
        best_center = (cx, cy)
        best_angle = angle
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
# EVOLVE-BLOCK-END
