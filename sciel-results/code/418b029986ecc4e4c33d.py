import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """
    Find min side length of outer hexagon (fixed angle) enclosing inner hexagons.
    Returns (cx, cy, side).
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0, 0, 0

    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # Constraint: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # Constraint: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            cx, cy, t = result.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx = sum(v[0] for v in all_vertices) / n_pts
    cy = sum(v[1] for v in all_vertices) / n_pts
    max_t = 0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            max_t = max(max_t, abs(proj))
    side = max_t / (math.sqrt(3) / 2)
    return cx, cy, side

def _check_disjoint(inner_hexes, eps=1e-6):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _hexagons_overlap(h1, h2, eps=1e-6):
    verts1 = _get_hex_vertices(h1[0], h1[1], h1[2])
    verts2 = _get_hex_vertices(h2[0], h2[1], h2[2])
    
    axes = []
    for verts in [verts1, verts2]:
        for k in range(6):
            x1, y1 = verts[k]
            x2, y2 = verts[(k + 1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            nx = -dy / length
            ny = dx / length
            axes.append((nx, ny))
            
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
            
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
            
    return True

def _optimize_inner_for_angle(initial_inner, target_angle, max_iter=200):
    """
    Optimize inner hexagon positions and angles to minimize outer side length
    for a FIXED outer angle.
    """
    def objective(params):
        if time.time() > _DEADLINE - 5.0:
            return 1e10
            
        inner = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner.append([x, y, a])
            
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Approximate distance check first for speed
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1.8: # Min dist is sqrt(3) ~ 1.732
                    if _hexagons_overlap(inner[i], inner[j], eps=-1e-4):
                        # Estimate penetration
                        # Use a simple penalty based on distance
                        min_dist = math.sqrt(3)
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2 * 100
                        else:
                            penalty += 100 # Overlap but dist > min_dist? Should not happen for convex
                            
        if penalty > 0:
            # Compute side length roughly to scale penalty
            cx, cy, side, _ = _min_enclosing_hex_side_lp(inner, target_angle)
            return side + penalty * 10
            
        cx, cy, side, _ = _min_enclosing_hex_side_lp(inner, target_angle)
        return side

    x0 = []
    for h in initial_inner:
        x0.extend([h[0], h[1], h[2]])
        
    bounds = [(-10, 10), (-10, 10), (-90, 90)] * N
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
        if result.success and result.fun < 1e9:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            return polished
    except Exception:
        pass
        
    return initial_inner

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate initial lattice configuration
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
            candidates.append((dist_sq, x, y, i, j))
            
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y, _, _ in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
            
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    best_side = float('inf')
    best_inner = [h[:] for h in inner]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a few outer angles
    angles_to_try = [0.0, 15.0, 30.0, 45.0]
    
    for angle in angles_to_try:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Optimize inner for this angle
        optimized_inner = _optimize_inner_for_angle(inner, angle, max_iter=100)
        
        # Get final side for this angle
        cx, cy, side = _min_enclosing_hex_side_lp(optimized_inner, angle)
        
        if side < best_side:
            best_side = side
            best_inner = optimized_inner
            best_center = (cx, cy)
            best_angle = angle
            
    # Final polish: refine outer angle around best_angle
    if time.time() < _DEADLINE - 10.0:
        for da in [x * 1.0 for x in range(-10, 11)]:
            test_angle = (best_angle + da) % 60
            cx, cy, side = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = test_angle
                
        # Fine refinement
        for da in [x * 0.1 for x in range(-10, 11)]:
            test_angle = (best_angle + da) % 60
            cx, cy, side = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = test_angle

    # Verify disjointness
    if not _check_disjoint(best_inner, eps=-1e-5):
        # If invalid, revert to initial lattice (which is valid)
        best_inner = inner
        best_center, best_side, best_angle = _min_enclosing_hex_side_lp(best_inner, 0.0)
        # Note: _min_enclosing_hex_side_lp returns cx, cy, side. 
        # We need to handle the return value correctly.
        cx, cy, side = _min_enclosing_hex_side_lp(best_inner, 0.0)
        best_center = (cx, cy)
        best_side = side
        best_angle = 0.0

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
