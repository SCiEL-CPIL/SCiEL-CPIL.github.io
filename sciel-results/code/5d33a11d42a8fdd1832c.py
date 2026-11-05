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
        # Rotate normal
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
        # Pre-compute rotated unit hex vertices (relative to center)
        pa = math.radians(pangle)
        pca, spa = math.cos(pa), math.sin(pa)
        for k in range(6):
            theta = math.radians(60 * k)
            ux = math.cos(theta)
            uy = math.sin(theta)
            rvx = pca * ux - spa * uy
            rvy = spa * ux + pca * uy
            vx = px + rvx
            vy = py + rvy
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint using vectorized SAT."""
    m = len(centers_angles)
    # Precompute vertices and normals for all hexagons
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    normals = [_hex_normals(p[2]) for p in centers_angles]
    
    for i in range(m):
        for j in range(i + 1, m):
            # Quick center-distance rejection
            dx = centers_angles[j][0] - centers_angles[i][0]
            dy = centers_angles[j][1] - centers_angles[i][1]
            if dx * dx + dy * dy > 4.5:
                continue
            # SAT check using precomputed normals
            axes = normals[i] + normals[j]
            sep = False
            for axis in axes:
                ax, ay = axis
                # Project polygon i
                vals_i = [vx * ax + vy * ay for vx, vy in polys[i]]
                minA, maxA = min(vals_i), max(vals_i)
                # Project polygon j
                vals_j = [vx * ax + vy * ay for vx, vy in polys[j]]
                minB, maxB = min(vals_j), max(vals_j)
                if maxA <= minB + tol or maxB <= minA + tol:
                    sep = True
                    break
            if not sep:
                return False
    return True

def _sdf_distance(p1, p2, a1, a2):
    """
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    Using support function approximation.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    dx_dir = dx / dist
    dy_dir = dy / dist
    
    def hex_support(angle_deg, dx_dir, dy_dir):
        """Support function of a unit regular hexagon in given direction."""
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
    
    # Radius of hex1 in direction of p2
    r1 = hex_support(a1, dx_dir, dy_dir)
    # Radius of hex2 in direction of p1 (opposite)
    r2 = hex_support(a2, -dx_dir, -dy_dir)
    
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
    
    def smooth_objective_lse(v, beta=50.0, beta_lse=80.0):
        """
        Smooth objective using log-sum-exp for outer side and SDF for overlaps.
        More differentiable for L-BFGS-B.
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        
        # Compute outer side with log-sum-exp smoothing
        a = math.radians(angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx, cy = center
        vals = []
        for (px, py, pangle) in cfg:
            pa = math.radians(pangle)
            pca, spa = math.cos(pa), math.sin(pa)
            for k in range(6):
                theta = math.radians(60 * k)
                ux = math.cos(theta)
                uy = math.sin(theta)
                rvx = pca * ux - spa * uy
                rvy = spa * ux + pca * uy
                vx = px + rvx
                vy = py + rvy
                for (nx, ny) in normals:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    vals.append(val)
        
        # Log-sum-exp for smooth max
        v_arr = np.array(vals)
        v_max = np.max(v_arr)
        if v_max < -50:
            outer_side = 0.0
        else:
            log_sum = math.log(math.fsum(math.exp(beta_lse * (v - v_max)) for v in v_arr))
            outer_side = v_max + log_sum / beta_lse
        
        # Compute overlap penalties using SDF with smooth max
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
    
    def lbfgs_polish(cfg_init, center_init, angle_init, time_budget):
        """L-BFGS-B optimization using smooth objective with LSE."""
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
        
        start_time = time.time()
        def callback(xk):
            return time.time() - start_time > (time_budget - 2.0)

        result = minimize(
            smooth_objective_lse,
            v0,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': 800, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10}
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
    
    # Time allocation: SA gets most time, then LBFGS polish
    sa_time_budget = TIME_LIMIT - 25.0  # leave 25s for LBFGS + safety
    
    while time.time() - t0 < sa_time_budget:
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
    
    # Local coordinate descent polish before L-BFGS
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 30.0:
        # Coordinate descent: try small perturbations on each parameter
        cd_time_budget = min(15.0, remaining_time - 20.0)
        cd_start = time.time()
        cur_cd_score = best_score
        cd_cfg = [row[:] for row in best_cfg]
        cd_center = best_center[:]
        cd_angle = best_angle
        
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for ss in step_sizes:
            if time.time() - cd_start > cd_time_budget:
                break
            improved = True
            while improved:
                if time.time() - cd_start > cd_time_budget:
                    break
                improved = False
                # Try moving each hexagon in 4 directions
                for idx in range(N):
                    for ddx, ddy in [(ss, 0), (-ss, 0), (0, ss), (0, -ss)]:
                        old_x, old_y = cd_cfg[idx][0], cd_cfg[idx][1]
                        cd_cfg[idx][0] += ddx
                        cd_cfg[idx][1] += ddy
                        s = full_score(cd_cfg, tuple(cd_center), cd_angle)
                        if s is not None and s < cur_cd_score:
                            cur_cd_score = s
                            improved = True
                        else:
                            cd_cfg[idx][0], cd_cfg[idx][1] = old_x, old_y
                # Try rotating each hexagon
                for idx in range(N):
                    for da in [ss * 30, -ss * 30]:
                        old_a = cd_cfg[idx][2]
                        cd_cfg[idx][2] += da
                        s = full_score(cd_cfg, tuple(cd_center), cd_angle)
                        if s is not None and s < cur_cd_score:
                            cur_cd_score = s
                            improved = True
                        else:
                            cd_cfg[idx][2] = old_a
                # Try moving outer center
                for ddx, ddy in [(ss, 0), (-ss, 0), (0, ss), (0, -ss)]:
                    old_cx, old_cy = cd_center[0], cd_center[1]
                    cd_center[0] += ddx
                    cd_center[1] += ddy
                    s = full_score(cd_cfg, tuple(cd_center), cd_angle)
                    if s is not None and s < cur_cd_score:
                        cur_cd_score = s
                        improved = True
                    else:
                        cd_center[0], cd_center[1] = old_cx, old_cy
                # Try rotating outer
                for da in [ss * 30, -ss * 30]:
                    old_a = cd_angle
                    cd_angle += da
                    s = full_score(cd_cfg, tuple(cd_center), cd_angle)
                    if s is not None and s < cur_cd_score:
                        cur_cd_score = s
                        improved = True
                    else:
                        cd_angle = old_a
        
        if cur_cd_score < best_score:
            best_score = cur_cd_score
            best_cfg = [row[:] for row in cd_cfg]
            best_center = cd_center[:]
            best_angle = cd_angle
    
    # L-BFGS-B polish using smooth objective with LSE
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            lbfgs_v, lbfgs_score = lbfgs_polish(
                best_cfg, best_center, best_angle, remaining_time - 3.0
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
    
    # Final high-precision polish with L-BFGS-B (tighter tolerance, LSE objective)
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
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
            
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback(xk):
                return time.time() - start_time > (remaining_time - 2.0)

            result = minimize(
                smooth_objective_lse,
                v0,
                method='L-BFGS-B',
                callback=callback,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            v_opt = result.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            center_opt = [v_opt[3*N], v_opt[3*N+1]]
            angle_opt = v_opt[3*N+2]
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    # Final safety check: ensure we're within time limit
    elapsed = time.time() - t0
    if elapsed > TIME_LIMIT - 1.0:
        pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
