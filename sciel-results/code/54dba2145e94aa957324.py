"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

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
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    Using SDF approximation via vertex-edge distance.
    """
    # For regular hexagons, the minimum distance can be approximated
    # by the distance between centers minus the sum of their "radii"
    # in the direction of the center-to-center vector.
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    # The "radius" of a regular hexagon in a given direction depends on the angle
    # relative to its orientation. For a unit hexagon with rotation a, the support
    # function in direction (dx, dy)/dist is:
    # h(a, theta) where theta is the angle of the direction relative to the hexagon
    
    def hex_radius(angle_deg, direction_deg):
        """
        Support function of a unit regular hexagon.
        The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
        (before rotation). The support function is the maximum of dot products.
        """
        a_rad = math.radians(angle_deg)
        d_rad = math.radians(direction_deg)
        dx_dir = math.cos(d_rad)
        dy_dir = math.sin(d_rad)
        
        max_val = 0.0
        for k in range(6):
            theta = math.radians(60 * k)
            vx = math.cos(theta)
            vy = math.sin(theta)
            # Rotate by angle_deg
            ca = math.cos(a_rad)
            sa = math.sin(a_rad)
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            dot = rvx * dx_dir + rvy * dy_dir
            if dot > max_val:
                max_val = dot
        return max_val
    
    # Direction from p1 to p2
    dir_deg = math.degrees(math.atan2(dy, dx))
    
    # Radius of hex1 in direction of p2
    r1 = hex_radius(a1, dir_deg)
    # Radius of hex2 in direction of p1 (opposite)
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
        # Central hex
        pts.append([0.0, 0.0, 0.0])
        # 6 around center at distance sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        # 5 more at larger radius
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
        return _outer_hex_side(cfg, center, angle)
    
    def smooth_objective(v, beta=50.0):
        """
        Smooth objective using soft-min for overlap constraints.
        v contains: [x1,y1,a1, ..., xN,yN,aN, cx,cy, cangle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        
        # Compute outer side
        outer_side = _outer_hex_side(cfg, tuple(center), angle)
        
        # Compute overlap penalties using SDF
        total_penalty = 0.0
        m = len(cfg)
        for i in range(m):
            for j in range(i + 1, m):
                d = _sdf_distance(cfg[i][0], cfg[i][1], cfg[i][2], 
                                  cfg[j][0], cfg[j][1], cfg[j][2])
                # Soft penalty for overlap (d < 0 means overlap)
                if d < 0:
                    total_penalty += math.exp(-beta * (-d))
        
        return outer_side + 10.0 * total_penalty
    
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
        step = 0.05
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
            if abs(f_max - f_best) < 1e-8:
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
    
    def lbfgs_polish(cfg_init, center_init, angle_init, time_budget):
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
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10}
        )
        
        v_opt = result.x
        cfg_opt, center_opt, angle_opt = unflatten(v_opt)
        
        # Check if the result is valid
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
    
    # Time allocation: SA gets most time, then NM polish
    nm_time_budget = TIME_LIMIT - 12.0  # leave 12s for NM + LBFGS + safety
    
    while time.time() - t0 < nm_time_budget:
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 3.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.08 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        else:
            # Perturb two hexagons
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
    
    # Nelder-Mead polish on the best configuration found
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 6.0:
        try:
            polished_v, polished_f = nelder_mead_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 6.0
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
    
    # L-BFGS-B polish using smooth objective
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 4.0:
        try:
            lbfgs_v, lbfgs_score = lbfgs_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 3.0
            )
            if lbfgs_score is not None and lbfgs_score < best_score:
                cfg_lbfgs = []
                for i in range(N):
                    cfg_lbfgs.append([lbfgs_v[3*i], lbfgs_v[3*i+1], lbfgs_v[3*i+2]])
                center_lbfgs = [lbfgs_v[3*N], lbfgs_v[3*N+1]]
                angle_lbfgs = lbfgs_v[3*N+2]
                
                # Verify with exact check
                verify_score = full_score(cfg_lbfgs, tuple(center_lbfgs), angle_lbfgs)
                if verify_score is not None and verify_score < best_score:
                    best_score = verify_score
                    best_cfg = cfg_lbfgs
                    best_center = center_lbfgs
                    best_angle = angle_lbfgs
        except Exception:
            pass
    
    # Final high-precision polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        try:
            # Run another NM with tighter parameters
            def tight_nm_polish(cfg_init, center_init, angle_init, time_budget):
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
                step = 0.01  # smaller step for tighter polish
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
                
                max_iters = 5000
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
            
            tight_v, tight_f = tight_nm_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 2.0
            )
            cfg_tight = []
            for i in range(N):
                cfg_tight.append([tight_v[3*i], tight_v[3*i+1], tight_v[3*i+2]])
            center_tight = [tight_v[3*N], tight_v[3*N+1]]
            angle_tight = tight_v[3*N+2]
            
            tight_score = full_score(cfg_tight, tuple(center_tight), angle_tight)
            if tight_score is not None and tight_score < best_score:
                best_score = tight_score
                best_cfg = cfg_tight
                best_center = center_tight
                best_angle = angle_tight
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
