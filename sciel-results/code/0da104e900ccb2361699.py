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

def _hex_normals(angle_deg):
    """
    Pre-computed edge normals for a unit hexagon with given rotation.
    The edges of a regular hexagon (vertices at 0, 60, ...) have normals at 30, 90, 150, 210, 270, 330.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

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

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    normals = [_hex_normals(p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            axes = normals[i] + normals[j]
            sep = False
            for axis in axes:
                minA, maxA = _project_polygon(polys[i], axis)
                minB, maxB = _project_polygon(polys[j], axis)
                if maxA <= minB + tol or maxB <= minA + tol:
                    sep = True
                    break
            if not sep:
                return False
    return True

def _hex_support(angle_deg, dx_dir, dy_dir):
    """Support function of a unit regular hexagon in a given direction."""
    a_rad = math.radians(angle_deg)
    ca = math.cos(a_rad)
    sa = math.sin(a_rad)
    max_val = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        vx = math.cos(theta)
        vy = math.sin(theta)
        rvx = ca * vx - sa * vy
        rvy = sa * vx + ca * vy
        dot = rvx * dx_dir + rvy * dy_dir
        if dot > max_val:
            max_val = dot
    return max_val

def _sdf_distance(p1x, p1y, a1, p2x, p2y, a2):
    """
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    """
    dx = p2x - p1x
    dy = p2y - p1y
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    dx_dir = dx / dist
    dy_dir = dy / dist
    
    r1 = _hex_support(a1, dx_dir, dy_dir)
    r2 = _hex_support(a2, -dx_dir, -dy_dir)
    
    return dist - r1 - r2

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_flower():
        """Flower pattern: 1 center + 6 inner ring + 5 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_dense_cluster():
        """Tight cluster arrangement."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_hexagonal_12():
        """12 hexagons: center + 5 inner + 6 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * i / 5 + math.pi/5
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    def smooth_objective(v, beta=60.0):
        """
        Smooth objective using soft penalty for overlap constraints.
        v contains: [x1,y1,a1, ..., xN,yN,aN, cx,cy, cangle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        
        outer_side = _outer_hex_side(cfg, tuple(center), angle)
        
        total_penalty = 0.0
        m = len(cfg)
        for i in range(m):
            for j in range(i + 1, m):
                d = _sdf_distance(cfg[i][0], cfg[i][1], cfg[i][2], 
                                  cfg[j][0], cfg[j][1], cfg[j][2])
                if d < 0:
                    total_penalty += math.exp(-beta * (-d))
        
        return outer_side + 10.0 * total_penalty
    
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
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster(), make_hexagonal_12()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-4, 4), random.uniform(-4, 4), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated Annealing phase
    T = 0.8
    T_end = 0.003
    alpha_sa = 0.9997
    
    sa_time_budget = TIME_LIMIT - 20.0  # leave 20s for LBFGS + safety
    
    while time.time() - t0 < sa_time_budget:
        r = random.random()
        
        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 3.0 * T)
        elif r < 0.75:
            old_cx, old_cy = center[0], center[1]
            step = 0.08 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.12 * (1.0 + 0.5 * T)
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
        
        T = max(T_end, T * alpha_sa)
    
    # L-BFGS-B polish with smooth objective
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback(xk):
                return time.time() - start_time > (remaining_time - 3.0)
            
            result = minimize(
                smooth_objective,
                v0,
                method='L-BFGS-B',
                callback=callback,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            v_opt = result.x
            cfg_opt, center_opt, angle_opt = unflatten(v_opt)
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    # Final high-precision polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback2(xk):
                return time.time() - start_time > (remaining_time - 2.0)
            
            result = minimize(
                smooth_objective,
                v0,
                method='L-BFGS-B',
                callback=callback2,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            v_opt = result.x
            cfg_opt, center_opt, angle_opt = unflatten(v_opt)
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
