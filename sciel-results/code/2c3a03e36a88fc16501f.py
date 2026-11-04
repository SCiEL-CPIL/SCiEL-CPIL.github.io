"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # safety margin below 150s

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
    # Quick reject: if center-to-center distance exceeds sum of circumradii, they can't overlap
    cxA = sum(v[0] for v in vertsA) / len(vertsA)
    cyA = sum(v[1] for v in vertsA) / len(vertsA)
    cxB = sum(v[0] for v in vertsB) / len(vertsB)
    cyB = sum(v[1] for v in vertsB) / len(vertsB)
    dist = math.hypot(cxA - cxB, cyA - cyB)
    if dist > 2.0 + 1e-9:  # circumradius of unit hexagon is 1.0
        return False
    
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

def _outer_hex_side_smooth(centers_angles, center, angle_deg, beta=5.0):
    """Smooth approximation of outer hex side using log-sum-exp."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    vals = []
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                vals.append(val)
    
    vals = np.array(vals)
    max_val = np.max(vals)
    if max_val < -100:
        return max_val
    shifted = vals - max_val
    lse = max_val + (1.0/beta) * math.log(np.sum(np.exp(beta * shifted)))
    return lse

def _collision_penalty_smooth(centers_angles, beta=5.0):
    """Smooth collision penalty using soft min of separation distances."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            axes = _axes_from_polygon(polys[i]) + _axes_from_polygon(polys[j])
            min_separation = float('inf')
            for axis in axes:
                minA, maxA = _project_polygon(polys[i], axis)
                minB, maxB = _project_polygon(polys[j], axis)
                sep = min(maxA, maxB) - max(minA, minB)
                if sep < min_separation:
                    min_separation = sep
            # If min_separation < 0, they overlap; penalize
            if min_separation < 0:
                penalty += min_separation ** 2
    return penalty

def _objective(v, beta=5.0):
    """Full objective: smooth outer side + collision penalty."""
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    outer_angle = v[3*N+2]
    
    outer_side = _outer_hex_side_smooth(cfg, center, outer_angle, beta)
    col_pen = _collision_penalty_smooth(cfg, beta)
    return outer_side + 5.0 * col_pen

def _flatten(cfg, center, angle):
    v = []
    for p in cfg:
        v.extend([p[0], p[1], p[2]])
    v.extend([center[0], center[1], angle])
    return np.array(v, dtype=float)

def _unflatten(v):
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = [v[3*N], v[3*N+1]]
    angle = v[3*N+2]
    return cfg, center, angle

def _make_hex_lattice():
    """3x4 hexagonal lattice."""
    pts = []
    row_height = math.sqrt(3) * 1.0
    col_width = 2.0 * 1.0
    for r in range(3):
        for c in range(4):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            pts.append([x - 3.0, y - row_height, 0.0])
    return pts

def _make_ring():
    """12 hexagons in a ring."""
    pts = []
    R = 2.5
    for i in range(N):
        theta = 2 * math.pi * i / N
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        angle_deg = math.degrees(theta)
        pts.append([x, y, angle_deg])
    return pts

def _make_grid_offset():
    """4x3 grid with offset."""
    pts = []
    for r in range(3):
        for c in range(4):
            x = c * 1.8 + (0.9 if r % 2 else 0.0)
            y = r * 1.6
            pts.append([x - 2.7, y - 1.6, 0.0])
    return pts

def _make_spiral():
    """Spiral arrangement."""
    pts = []
    for i in range(N):
        theta = i * 0.8
        r = 0.5 + i * 0.25
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        angle_deg = math.degrees(theta) * 0.5
        pts.append([x, y, angle_deg])
    return pts

def construct():
    t0 = time.time()
    random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    # Generate multiple initial configurations
    inits = [
        _make_hex_lattice(),
        _make_ring(),
        _make_grid_offset(),
        _make_spiral(),
    ]
    
    # Add random perturbations of each init
    all_inits = []
    for base in inits:
        all_inits.append(base)
        for _ in range(3):
            perturbed = []
            for p in base:
                perturbed.append([
                    p[0] + random.gauss(0, 0.5),
                    p[1] + random.gauss(0, 0.5),
                    p[2] + random.gauss(0, 10.0)
                ])
            all_inits.append(perturbed)
    
    # Also try purely random configs
    for _ in range(4):
        cfg = []
        for i in range(N):
            cfg.append([
                random.uniform(-3, 3),
                random.uniform(-3, 3),
                random.uniform(0, 60)
            ])
        all_inits.append(cfg)
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    # Evaluate all inits
    for init in all_inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        # Fallback
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    # Simulated annealing phase
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    T = 1.0
    T_end = 0.005
    alpha = 0.9995
    
    # Time budget allocation
    sa_time_limit = TIME_LIMIT - 30.0  # reserve 30s for L-BFGS polish
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.3 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 8.0 + 5.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.15 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 3.0 + 3.0 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.2 * (1.0 + 0.5 * T)
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
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.85:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            # Invalid: revert
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha)
    
    # L-BFGS-B polish phase
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        # Try polishing from the best SA result
        v_init = _flatten(best_cfg, best_center, best_angle)
        
        def obj_and_grad(v):
            f = _objective(v, beta=8.0)
            # Numerical gradient
            h = 1e-6
            grad = np.zeros_like(v)
            for i in range(len(v)):
                v_plus = v.copy()
                v_plus[i] += h
                f_plus = _objective(v_plus, beta=8.0)
                grad[i] = (f_plus - f) / h
            return f, grad
        
        try:
            res = minimize(
                lambda v: _objective(v, beta=8.0),
                v_init,
                jac=lambda v: np.array([( _objective(v+h*np.eye(len(v))[i], beta=8.0) - _objective(v, beta=8.0)) / h for i, h in enumerate([1e-6]*len(v))]),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                cfg_polished, center_polished, angle_polished = _unflatten(res.x)
                polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
                if polished_score is not None and polished_score < best_score:
                    best_score = polished_score
                    best_cfg = cfg_polished
                    best_center = center_polished
                    best_angle = angle_polished
        except Exception:
            pass
    
    # Final high-precision polish if time remains
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        v_init = _flatten(best_cfg, best_center, best_angle)
        try:
            res2 = minimize(
                lambda v: _objective(v, beta=15.0),
                v_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.success:
                cfg_polished2, center_polished2, angle_polished2 = _unflatten(res2.x)
                polished_score2 = full_score(cfg_polished2, tuple(center_polished2), angle_polished2)
                if polished_score2 is not None and polished_score2 < best_score:
                    best_score = polished_score2
                    best_cfg = cfg_polished2
                    best_center = center_polished2
                    best_angle = angle_polished2
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
