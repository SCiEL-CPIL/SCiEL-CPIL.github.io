"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 118.0  # safety margin below 120s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _sdf_hex_point(px, py, cx, cy, angle_deg):
    """Signed distance from point (px,py) to hexagon centered at (cx,cy) with rotation angle_deg.
    Positive = outside, negative = inside."""
    # Transform point to hexagon's local frame
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # For a unit hexagon, the apothem is sqrt(3)/2
    apothem = math.sqrt(3) / 2.0
    
    # Check if point is inside by checking all 6 edge constraints
    inside = True
    min_dist = float('inf')
    
    for k in range(6):
        theta = math.radians(60 * k)
        # Edge normal direction
        nx = math.cos(theta + math.pi / 2)
        ny = math.sin(theta + math.pi / 2)
        # Edge midpoint
        mx = 0.5 * (math.cos(theta) + math.cos(theta + math.radians(60)))
        my = 0.5 * (math.sin(theta) + math.sin(theta + math.radians(60)))
        # Actually, let's use a simpler approach: check distance to each edge
        # Edge k connects vertex k and vertex k+1
        v1x = math.cos(math.radians(60 * k))
        v1y = math.sin(math.radians(60 * k))
        v2x = math.cos(math.radians(60 * (k + 1)))
        v2y = math.sin(math.radians(60 * (k + 1)))
        
        # Distance from point to line through v1 and v2
        ex = v2x - v1x
        ey = v2y - v1y
        len_e = math.hypot(ex, ey)
        if len_e < 1e-12:
            continue
        # Normal to edge
        nx = -ey / len_e
        ny = ex / len_e
        # Signed distance
        dist = (lx - v1x) * nx + (ly - v1y) * ny
        if dist > apothem + 1e-9:
            inside = False
            break
        if dist > 0:
            min_dist = min(min_dist, dist)
    
    if inside:
        return -min_dist if min_dist < float('inf') else 0.0
    else:
        return min_dist

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2):
    """Approximate signed distance between two hexagons.
    Returns negative if overlapping, positive if separated."""
    # Sample points on the boundary of hex1 and find minimum distance to hex2
    min_dist = float('inf')
    
    # Sample 24 points on hex1 boundary
    for i in range(24):
        theta = 2 * math.pi * i / 24
        # Point on hex1 boundary (approximate)
        px = cx1 + math.cos(theta)
        py = cy1 + math.sin(theta)
        d = _sdf_hex_point(px, py, cx2, cy2, a2)
        if d < min_dist:
            min_dist = d
    
    # Also check center-to-center distance as a lower bound
    center_dist = math.hypot(cx1 - cx2, cy1 - cy2)
    
    # The minimum distance between hexagons is at most the center distance minus some offset
    # But we want the actual gap. If min_dist > 0, they're separated by at least min_dist
    # If min_dist < 0, they overlap
    
    return min_dist

def _smooth_collision_penalty(cfg, strength=100.0):
    """Compute smooth penalty for overlapping hexagons."""
    penalty = 0.0
    m = len(cfg)
    for i in range(m):
        for j in range(i + 1, m):
            x1, y1, a1 = cfg[i]
            x2, y2, a2 = cfg[j]
            d = _sdf_hex_hex(x1, y1, a1, x2, y2, a2)
            if d < 0:
                penalty += strength * (-d) ** 2
    return penalty

def _objective(params, center, angle):
    """Objective: outer side length + collision penalty."""
    cfg = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        cfg.append((x, y, a))
    
    side = _outer_hex_side(cfg, center, angle)
    penalty = _smooth_collision_penalty(cfg, strength=50.0)
    
    # Check validity
    if not _is_valid(cfg):
        return side + penalty + 100.0
    return side + penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_initial():
        """Hexagonal lattice arrangement: 3 rows x 4 columns."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    cfg = make_initial()
    center = [0.0, 0.0]
    angle = 0.0
    
    cur_score = full_score(cfg, tuple(center), angle)
    if cur_score is None:
        for p in cfg:
            p[0] *= 1.5
            p[1] *= 1.5
        cur_score = full_score(cfg, tuple(center), angle)
        if cur_score is None:
            cur_score = 40.0
            for i, p in enumerate(cfg):
                p[0] = (i % 4 - 1.5) * 3.0
                p[1] = (i // 4 - 1.0) * 3.0
    
    best_score = cur_score
    best_cfg = [row[:] for row in cfg]
    best_center = center[:]
    best_angle = angle
    
    # Phase 1: Differential Evolution for global optimization
    elapsed = time.time() - t0
    de_budget = min(60.0, TIME_LIMIT - elapsed - 20.0)
    
    if de_budget > 10.0:
        # Define bounds for each hexagon: x, y in [-5, 5], angle in [0, 360)
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((0.0, 360.0))
        
        def de_objective(params):
            cfg_local = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                cfg_local.append((x, y, a))
            
            if not _is_valid(cfg_local):
                return 100.0 + _smooth_collision_penalty(cfg_local, strength=20.0)
            
            side = _outer_hex_side(cfg_local, tuple(center), angle)
            return side
        
        # Use differential evolution
        try:
            result = differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            if result.success or result.fun < best_score:
                params = result.x
                new_cfg = []
                for i in range(N):
                    new_cfg.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
                
                new_score = full_score(new_cfg, tuple(center), angle)
                if new_score is not None and new_score < best_score:
                    best_score = new_score
                    best_cfg = new_cfg
                    cfg = new_cfg
                    cur_score = new_score
        except Exception:
            pass
    
    # Phase 2: Local polish with L-BFGS-B
    elapsed = time.time() - t0
    polish_budget = min(30.0, TIME_LIMIT - elapsed - 5.0)
    
    if polish_budget > 5.0:
        # Optimize all parameters: positions, angles, center, outer angle
        def polish_objective(params):
            cfg_local = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                cfg_local.append((x, y, a))
            
            cx = params[N * 3]
            cy = params[N * 3 + 1]
            outer_angle = params[N * 3 + 2]
            
            if not _is_valid(cfg_local):
                return 100.0 + _smooth_collision_penalty(cfg_local, strength=50.0)
            
            side = _outer_hex_side(cfg_local, (cx, cy), outer_angle)
            return side
        
        x0 = []
        for p in cfg:
            x0.extend([p[0], p[1], p[2]])
        x0.extend([center[0], center[1], angle])
        
        try:
            result = minimize(
                polish_objective,
                x0=x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                params = result.x
                new_cfg = []
                for i in range(N):
                    new_cfg.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
                new_center = [params[N * 3], params[N * 3 + 1]]
                new_angle = params[N * 3 + 2]
                
                new_score = full_score(new_cfg, tuple(new_center), new_angle)
                if new_score is not None and new_score < best_score:
                    best_score = new_score
                    best_cfg = new_cfg
                    best_center = new_center
                    best_angle = new_angle
        except Exception:
            pass
    
    # Phase 3: Simulated annealing for fine-tuning
    elapsed = time.time() - t0
    sa_budget = max(5.0, TIME_LIMIT - elapsed - 2.0)
    
    if sa_budget > 2.0:
        cfg = [row[:] for row in best_cfg]
        center = best_center[:]
        angle = best_angle
        cur_score = best_score
        
        T = 0.1
        T_end = 0.001
        alpha = 0.999
        
        while time.time() - t0 < TIME_LIMIT - 2.0:
            r = random.random()
            if r < 0.5:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                cfg[idx][0] += random.gauss(0, 0.1)
                cfg[idx][1] += random.gauss(0, 0.1)
            elif r < 0.7:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 3.0)
            elif r < 0.85:
                old_cx, old_cy = center[0], center[1]
                center[0] += random.gauss(0, 0.05)
                center[1] += random.gauss(0, 0.05)
            else:
                old_angle = angle
                angle += random.gauss(0, 1.0)
            
            new_score = full_score(cfg, tuple(center), angle)
            
            if new_score is not None:
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
                else:
                    if r < 0.5:
                        cfg[idx][0], cfg[idx][1] = old_x, old_y
                    elif r < 0.7:
                        cfg[idx][2] = old_ang
                    elif r < 0.85:
                        center[0], center[1] = old_cx, old_cy
                    else:
                        angle = old_angle
            else:
                if r < 0.5:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.7:
                    cfg[idx][2] = old_ang
                elif r < 0.85:
                    center[0], center[1] = old_cx, old_cy
                else:
                    angle = old_angle
            
            T = max(T_end, T * alpha)
    
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, (best_center[0], best_center[1]), best_score, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
