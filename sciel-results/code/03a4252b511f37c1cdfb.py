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

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
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

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _sdf_distance(p1, p2, a1, a2):
    """
    Compute the signed distance between two unit hexagons using support function.
    Positive means separated, negative means overlapping.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    def hex_radius(angle_deg, direction_deg):
        a_rad = math.radians(angle_deg)
        d_rad = math.radians(direction_deg)
        dx_dir = math.cos(d_rad)
        dy_dir = math.sin(d_rad)
        
        max_val = 0.0
        ca = math.cos(a_rad)
        sa = math.sin(a_rad)
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
    
    dir_deg = math.degrees(math.atan2(dy, dx))
    r1 = hex_radius(a1, dir_deg)
    r2 = hex_radius(a2, dir_deg + 180)
    
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
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    def smooth_objective(v, beta=30.0):
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
        
        return outer_side + 5.0 * total_penalty
    
    def nm_polish(cfg_init, center_init, angle_init, time_budget, use_smooth=True):
        """Nelder-Mead polish."""
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
        
        if use_smooth:
            def objective(v):
                return smooth_objective(v)
        else:
            def objective(v):
                cfg, center, angle = unflatten(v)
                s = full_score(cfg, tuple(center), angle)
                if s is None:
                    return 1e6
                return s
        
        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        simplex = [v0[:]]
        step = 0.05 if use_smooth else 0.01
        for i in range(n_params):
            vi = v0[:]
            vi[i] += step
            simplex.append(vi)
        
        alpha_nm = 1.0
        gamma_nm = 2.0
        rho_nm = 0.5
        sigma_nm = 0.5
        
        best_f = min(objective(v) for v in simplex)
        best_v = min(simplex, key=objective)
        
        max_iters = 5000 if use_smooth else 8000
        iter_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            simplex.sort(key=lambda v: objective(v))
            f_best = objective(simplex[0])
            
            if f_best < best_f:
                best_f = f_best
                best_v = simplex[0][:]
            
            f_max = objective(simplex[-1])
            if abs(f_max - f_best) < 1e-10:
                break
            
            centroid = [sum(v[i] for v in simplex[:-1]) / n_params for i in range(n_params)]
            
            xr = [centroid[i] + alpha_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
            fr = objective(xr)
            
            if fr < f_best:
                xe = [centroid[i] + gamma_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif fr < objective(simplex[-2]):
                simplex[-1] = xr
            else:
                if fr < objective(simplex[-1]):
                    xc = [centroid[i] + rho_nm * (simplex[-1][i] - centroid[i]) for i in range(n_params)]
                else:
                    xc = [centroid[i] + rho_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fc = objective(xc)
                if fc < min(fr, objective(simplex[-1])):
                    simplex[-1] = xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = [simplex[0][j] + sigma_nm * (simplex[i][j] - simplex[0][j]) for j in range(n_params)]
            
            iter_count += 1
        
        return best_v, best_f
    
    def lbfgs_smooth(cfg_init, center_init, angle_init, time_budget):
        """L-BFGS-B optimization using smooth objective."""
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
        
        v0 = flatten(cfg_init, center_init, angle_init)
        
        result = minimize(
            smooth_objective,
            v0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        v_opt = result.x
        cfg_opt, center_opt, angle_opt = unflatten(v_opt)
        
        score = full_score(cfg_opt, tuple(center_opt), angle_opt)
        if score is not None:
            return v_opt, score
        return v0, None
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster()]
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
    
    sa_time_budget = TIME_LIMIT - 30.0  # leave 30s for polishers
    
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
    
    # Smooth NM polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 18.0:
        try:
            v, f = nm_polish(best_cfg, best_center, best_angle, TIME_LIMIT - 15.0, use_smooth=True)
            cfg_p = [v[3*i], v[3*i+1], v[3*i+2] for i in range(N)]
            c_p = [v[3*N], v[3*N+1]]
            a_p = v[3*N+2]
            s_p = full_score(cfg_p, tuple(c_p), a_p)
            if s_p is not None and s_p < best_score:
                best_score = s_p
                best_cfg = cfg_p
                best_center = c_p
                best_angle = a_p
        except Exception:
            pass
    
    # L-BFGS-B polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            lbfgs_v, lbfgs_score = lbfgs_smooth(best_cfg, best_center, best_angle, TIME_LIMIT - 7.0)
            if lbfgs_score is not None and lbfgs_score < best_score:
                cfg_l = [lbfgs_v[3*i], lbfgs_v[3*i+1], lbfgs_v[3*i+2] for i in range(N)]
                c_l = [lbfgs_v[3*N], lbfgs_v[3*N+1]]
                a_l = lbfgs_v[3*N+2]
                verify_score = full_score(cfg_l, tuple(c_l), a_l)
                if verify_score is not None and verify_score < best_score:
                    best_score = verify_score
                    best_cfg = cfg_l
                    best_center = c_l
                    best_angle = a_l
        except Exception:
            pass
    
    # Final exact NM polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            v, f = nm_polish(best_cfg, best_center, best_angle, TIME_LIMIT - 2.0, use_smooth=False)
            cfg_t = [v[3*i], v[3*i+1], v[3*i+2] for i in range(N)]
            c_t = [v[3*N], v[3*N+1]]
            a_t = v[3*N+2]
            s_t = full_score(cfg_t, tuple(c_t), a_t)
            if s_t is not None and s_t < best_score:
                best_score = s_t
                best_cfg = cfg_t
                best_center = c_t
                best_angle = a_t
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
