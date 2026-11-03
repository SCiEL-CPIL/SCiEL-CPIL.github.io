"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 110.0  # safety margin below 120s

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
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
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

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_initial():
        """Hexagonal lattice arrangement: rows of 4, 4, 4 with offset."""
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
    
    def make_ring_initial():
        """12 hexagons arranged in a ring around the center."""
        pts = []
        R = 2.5  # radius of ring
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            # Orient each hexagon to face outward
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_grid_initial():
        """Grid-like arrangement with more spacing."""
        pts = []
        # 3 rows x 4 cols with hexagonal offset
        row_h = math.sqrt(3)
        col_w = 2.0
        for r in range(3):
            for c in range(4):
                x = c * col_w + (col_w / 2 if r % 2 else 0.0)
                y = r * row_h
                pts.append([x - 3.0, y - row_h, 0.0])
        return pts
    
    def nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
        """Nelder-Mead optimization to polish the configuration."""
        # Flatten all parameters into a single vector
        # 12 hexagons * 3 params (x, y, angle) + 2 center + 1 outer angle = 39 params
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return v
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle
        
        def objective(v):
            cfg, center, angle = unflatten(v)
            s = full_score(cfg, tuple(center), angle)
            if s is None:
                return 1e6  # Large penalty for invalid
            return s
        
        # Initial simplex
        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        simplex = [v0[:]]
        step = 0.1
        for i in range(n_params):
            vi = v0[:]
            vi[i] += step
            simplex.append(vi)
        
        # Nelder-Mead iterations
        alpha_nm = 1.0   # reflection
        gamma_nm = 2.0   # expansion
        rho_nm = 0.5     # contraction
        sigma_nm = 0.5   # shrink
        
        best_f = min(objective(v) for v in simplex)
        best_v = min(simplex, key=objective)
        
        max_iters = 2000
        iter_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            # Sort simplex by objective value
            simplex.sort(key=lambda v: objective(v))
            f_best = objective(simplex[0])
            
            if f_best < best_f:
                best_f = f_best
                best_v = simplex[0][:]
            
            # Check convergence
            f_max = objective(simplex[-1])
            if abs(f_max - f_best) < 1e-7:
                break
            
            # Centroid of all points except the worst
            centroid = [sum(v[i] for v in simplex[:-1]) / n_params for i in range(n_params)]
            
            # Reflection
            xr = [centroid[i] + alpha_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
            fr = objective(xr)
            
            if fr < f_best:
                # Expansion
                xe = [centroid[i] + gamma_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif fr < objective(simplex[-2]):
                simplex[-1] = xr
            else:
                # Contraction
                if fr < objective(simplex[-1]):
                    # Outside contraction
                    xc = [centroid[i] + rho_nm * (simplex[-1][i] - centroid[i]) for i in range(n_params)]
                else:
                    # Inside contraction
                    xc = [centroid[i] + rho_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fc = objective(xc)
                if fc < min(fr, objective(simplex[-1])):
                    simplex[-1] = xc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = [simplex[0][j] + sigma_nm * (simplex[i][j] - simplex[0][j]) for j in range(n_params)]
            
            iter_count += 1
        
        return best_v, best_f
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_initial(), make_ring_initial(), make_grid_initial()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        # Fallback: spread out widely
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    T = 0.8
    T_end = 0.003
    alpha = 0.9997
    
    # Reserve time for Nelder-Mead polish
    nm_time_budget = 10.0  # seconds reserved for polishing
    sa_time_limit = TIME_LIMIT - nm_time_budget
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.50:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.70:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.80:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.90:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
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
                # Revert
                if r < 0.50:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.70:
                    cfg[idx][2] = old_ang
                elif r < 0.80:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.90:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            # Invalid: revert
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha)
    
    # Nelder-Mead polish on the best configuration found
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:  # Only polish if we have enough time
        try:
            polished_v, polished_f = nelder_mead_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 2.0
            )
            # Unflatten the polished result
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            # Verify the polished result
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass  # Keep the SA result if polish fails
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
