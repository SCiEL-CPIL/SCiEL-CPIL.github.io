"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 148.0  # safety margin below 150s
APOTHEM = math.sqrt(3) / 2.0

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
                val = (nx * (vx - cx) + ny * (vy - cy)) / APOTHEM
                if val > max_need:
                    max_need = val
    return max_need

def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at origin with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * APOTHEM
    
    # Precompute edge normals
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Check if inside
    max_proj = 0.0
    inside = True
    for nx, ny in normals:
        proj = nx * lx + ny * ly
        if proj > max_proj:
            max_proj = proj
        if proj > apothem + 1e-12:
            inside = False
    
    if inside:
        return -(apothem - max_proj)
    else:
        # Outside: compute distance to nearest point on polygon
        min_d = float('inf')
        for k in range(6):
            t1 = math.radians(60 * k)
            t2 = math.radians(60 * (k + 1))
            v1 = (side * math.cos(t1), side * math.sin(t1))
            v2 = (side * math.cos(t2), side * math.sin(t2))
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((lx - v1[0]) * dx + (ly - v1[1]) * dy) / len2))
            qx, qy = v1[0] + t * dx, v1[1] + t * dy
            d = math.hypot(lx - qx, ly - qy)
            if d < min_d:
                min_d = d
        return min_d

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            # Quick rejection: if center distance is too large, they can't overlap
            dx = centers_angles[i][0] - centers_angles[j][0]
            dy = centers_angles[i][1] - centers_angles[j][1]
            if dx * dx + dy * dy > 4.01:  # (2.005)^2
                continue
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _smooth_collision_penalty(cfg, sigma=0.05):
    """Smooth penalty for collisions using SDF between hexagons."""
    m = len(cfg)
    total_penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            cx1, cy1, a1 = cfg[i][0], cfg[i][1], cfg[i][2]
            cx2, cy2, a2 = cfg[j][0], cfg[j][1], cfg[j][2]
            
            # Sample vertices of hex1 and compute distance to hex2
            verts1 = _hex_vertices(cx1, cy1, a1)
            verts2 = _hex_vertices(cx2, cy2, a2)
            min_d = float('inf')
            for (vx, vy) in verts1:
                d = _sdf_hex_point(vx - cx2, vy - cy2, a2)
                if d < min_d:
                    min_d = d
            for (vx, vy) in verts2:
                d = _sdf_hex_point(vx - cx1, vy - cy1, a1)
                if d < min_d:
                    min_d = d
            
            if min_d < 2.0 * sigma:
                penalty = math.exp(-min_d / sigma) - 1.0
                total_penalty += penalty
    return total_penalty

def _smooth_objective(v):
    """Smooth objective for optimization: outer side + collision penalty."""
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    s = _outer_hex_side(cfg, center, angle)
    penalty = _smooth_collision_penalty(cfg, sigma=0.03)
    return s + 20.0 * penalty

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
    
    def make_cluster():
        """Tight cluster: central hex + ring of 6 + outer 5."""
        pts = []
        # Central
        pts.append([0.0, 0.0, 0.0])
        # Inner ring of 6
        R1 = 2.0 * APOTHEM
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        # Outer 5
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_cluster()]
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
    
    # Differential Evolution for global search
    def de_objective(v):
        cfg_de = []
        for i in range(N):
            cfg_de.append([v[3*i], v[3*i+1], v[3*i+2]])
        c_de = (v[3*N], v[3*N+1])
        a_de = v[3*N+2]
        s = _outer_hex_side(cfg_de, c_de, a_de)
        penalty = _smooth_collision_penalty(cfg_de, sigma=0.05)
        return s + 30.0 * penalty
    
    # Define bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-1, 1))  # center x
    bounds.append((-1, 1))  # center y
    bounds.append((-90, 90))  # outer angle
    
    # Run differential evolution
    elapsed = time.time() - t0
    de_time_budget = min(40.0, TIME_LIMIT - 60.0)  # Leave time for polishing
    
    if elapsed < de_time_budget:
        try:
            res_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 0.9),
                recombination=0.7,
                seed=42,
                init='sobol',
                polish=False
            )
            v_de = res_de.x
            cfg_de = []
            for i in range(N):
                cfg_de.append([v_de[3*i], v_de[3*i+1], v_de[3*i+2]])
            center_de = [v_de[3*N], v_de[3*N+1]]
            angle_de = v_de[3*N+2]
            
            de_score = full_score(cfg_de, tuple(center_de), angle_de)
            if de_score is not None and de_score < best_score:
                best_score = de_score
                best_cfg = cfg_de
                best_center = center_de
                best_angle = angle_de
        except Exception:
            pass
    
    # Simulated Annealing phase
    T = 0.5
    T_end = 0.005
    alpha_sa = 0.9995
    
    elapsed = time.time() - t0
    sa_time_budget = min(40.0, TIME_LIMIT - 30.0)
    
    if elapsed < sa_time_budget:
        while time.time() - t0 < sa_time_budget:
            r = random.random()
            
            if r < 0.45:
                # Move one hexagon
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                step = 0.1 * (1.0 + 0.5 * T)
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
            elif r < 0.65:
                # Rotate one hexagon
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 3.0 + 2.0 * T)
            elif r < 0.75:
                # Move center
                old_cx, old_cy = center[0], center[1]
                step = 0.05 * (1.0 + 0.5 * T)
                center[0] += random.gauss(0, step)
                center[1] += random.gauss(0, step)
            elif r < 0.85:
                # Rotate outer
                old_angle = angle
                angle += random.gauss(0, 1.0 + 1.0 * T)
            else:
                # Perturb two hexagons
                idx1, idx2 = random.sample(range(N), 2)
                old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
                old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
                step = 0.08 * (1.0 + 0.5 * T)
                cfg[idx1][0] += random.gauss(0, step)
                cfg[idx1][1] += random.gauss(0, step)
                cfg[idx2][0] += random.gauss(0, step)
                cfg[idx2][1] += random.gauss(0, step)
            
            # Use fast smooth objective for SA
            s_val = _outer_hex_side(cfg, tuple(center), angle)
            p_val = _smooth_collision_penalty(cfg, sigma=0.05)
            new_score = s_val + 20.0 * p_val
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                # Check if this is a valid configuration to update best_score
                valid_score = full_score(cfg, tuple(center), angle)
                if valid_score is not None and valid_score < best_score:
                    best_score = valid_score
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
            
            T = max(T_end, T * alpha_sa)
    
    # L-BFGS-B polish with smooth objective
    elapsed = time.time() - t0
    if elapsed < TIME_LIMIT - 15.0:
        try:
            v0 = []
            for p in best_cfg:
                v0.extend([p[0], p[1], p[2]])
            v0.extend([best_center[0], best_center[1], best_angle])
            
            res = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            
            v_lbfgs = res.x
            cfg_lbfgs = []
            for i in range(N):
                cfg_lbfgs.append([v_lbfgs[3*i], v_lbfgs[3*i+1], v_lbfgs[3*i+2]])
            center_lbfgs = [v_lbfgs[3*N], v_lbfgs[3*N+1]]
            angle_lbfgs = v_lbfgs[3*N+2]
            
            lbfgs_score = full_score(cfg_lbfgs, tuple(center_lbfgs), angle_lbfgs)
            if lbfgs_score is not None and lbfgs_score < best_score:
                best_score = lbfgs_score
                best_cfg = cfg_lbfgs
                best_center = center_lbfgs
                best_angle = angle_lbfgs
        except Exception:
            pass
    
    # Nelder-Mead polish
    def nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
        """Nelder-Mead optimization to polish the configuration."""
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
                return 1e6
            return s
        
        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        simplex = [v0[:]]
        step = 0.03
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
        
        max_iters = 3000
        iter_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            simplex.sort(key=lambda v: objective(v))
            f_best = objective(simplex[0])
            
            if f_best < best_f:
                best_f = f_best
                best_v = simplex[0][:]
            
            f_max = objective(simplex[-1])
            if abs(f_max - f_best) < 1e-9:
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
    
    elapsed = time.time() - t0
    if elapsed < TIME_LIMIT - 10.0:
        try:
            polished_v, polished_f = nelder_mead_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 5.0
            )
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    # Second L-BFGS-B pass for additional refinement
    elapsed = time.time() - t0
    if elapsed < TIME_LIMIT - 5.0:
        try:
            v0 = []
            for p in best_cfg:
                v0.extend([p[0], p[1], p[2]])
            v0.extend([best_center[0], best_center[1], best_angle])
            
            res = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 5000}
            )
            
            v_lbfgs2 = res.x
            cfg_lbfgs2 = []
            for i in range(N):
                cfg_lbfgs2.append([v_lbfgs2[3*i], v_lbfgs2[3*i+1], v_lbfgs2[3*i+2]])
            center_lbfgs2 = [v_lbfgs2[3*N], v_lbfgs2[3*N+1]]
            angle_lbfgs2 = v_lbfgs2[3*N+2]
            
            lbfgs2_score = full_score(cfg_lbfgs2, tuple(center_lbfgs2), angle_lbfgs2)
            if lbfgs2_score is not None and lbfgs2_score < best_score:
                best_score = lbfgs2_score
                best_cfg = cfg_lbfgs2
                best_center = center_lbfgs2
                best_angle = angle_lbfgs2
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
