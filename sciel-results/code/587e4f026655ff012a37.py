"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

# Pre-compute base hexagon vertices (side=1, angle=0)
# Vertices at angles 0, 60, 120, 180, 240, 300
_BASE_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

def _hex_vertices_array(cx, cy, angle_deg):
    """
    Returns (6, 2) array of vertices for a hexagon.
    angle_deg is in degrees.
    """
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ _BASE_VERTS.T).T + np.array([cx, cy])

def _check_pair_collision(verts1, verts2):
    """
    Fast SAT check for two convex hexagons represented as (6,2) arrays.
    Returns True if they collide (overlap).
    """
    # Combine vertices to find all potential separating axes
    # For convex polygons, we only need to check normals of edges.
    # verts1 has 6 edges, verts2 has 6 edges.
    
    def get_normals(verts):
        # verts is (6,2)
        edges = np.roll(verts, -1, axis=0) - verts
        # normals are perpendicular to edges
        # if edge is (dx, dy), normal is (-dy, dx)
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        # Normalize normals (length doesn't matter for dot product sign, 
        # but helps if we were comparing magnitudes, here we just project)
        # Actually, for SAT, we just project onto the normal direction.
        # We don't strictly need to normalize if we just check overlap of intervals,
        # but normalizing makes the "gap" in units of distance.
        # However, for boolean overlap, unnormalized is fine as long as consistent.
        # Let's normalize to be safe for numerical stability in projection.
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return normals / lengths

    n1 = get_normals(verts1)
    n2 = get_normals(verts2)
    
    all_normals = np.vstack((n1, n2))
    
    # Project verts1 and verts2 onto all_normals
    # verts1: (6, 2), all_normals: (12, 2)
    # proj1: (12, 6)
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)
    
    # Check for separation on any axis
    # Separation if max1 < min2 OR max2 < min1
    # Overlap if NOT (max1 < min2 OR max2 < min1)
    # i.e. Overlap if max1 >= min2 AND max2 >= min1
    
    # We want to know if they COLLIDE.
    # Collision = No separating axis found.
    # So if ANY axis separates them, return False (no collision).
    
    separates = (max1 < min2) | (max2 < min1)
    
    if np.any(separates):
        return False
    return True

def _is_valid_fast(cfg):
    """
    cfg: list of [x, y, angle_deg]
    Returns True if all pairs are disjoint.
    """
    # Pre-compute all vertices
    verts_list = []
    for p in cfg:
        verts_list.append(_hex_vertices_array(p[0], p[1], p[2]))
    
    # Distance pre-filter
    # Max distance between centers of two non-colliding unit hexagons?
    # If distance > 4, they definitely don't touch.
    # Actually, for unit hexagons, the "radius" to vertex is 1.
    # Max extent is 1. So if dist > 2, they are separated by at least dist-2.
    # Wait, hexagon width is 2. Height is sqrt(3).
    # If center distance > 2, they might still touch if aligned?
    # No, if center distance > 2, the closest points are at least dist - 1 - 1 = dist - 2.
    # If dist > 2, they are separated.
    # Let's use a slightly tighter bound to be safe, e.g., 2.0.
    # Actually, the "circumradius" is 1. So if dist > 2, they don't touch.
    
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, _ = cfg[i]
            xj, yj, _ = cfg[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.0: # dist > 2
                continue
            if _check_pair_collision(verts_list[i], verts_list[j]):
                return False
    return True

def _outer_hex_side_fast(cfg, center, angle_deg):
    """
    Compute minimum side length S of outer hexagon containing all inner hexagons.
    cfg: list of [x, y, angle_deg]
    center: (cx, cy)
    angle_deg: rotation of outer hexagon
    """
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    
    # 6 outward normals of the outer hexagon
    # Outer hexagon is regular. Normals are perpendicular to edges.
    # If outer hexagon is rotated by a, its edges are rotated by a.
    # Normals are at a + 90 + 60*k.
    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    
    cx, cy = center
    
    max_need = 0.0
    
    for p in cfg:
        verts = _hex_vertices_array(p[0], p[1], p[2])
        # Project verts onto normals
        # verts: (6,2), normals: (6,2)
        # We want dot product of (v - c) with n
        # (v - c) @ n
        rel_verts = verts - np.array([cx, cy])
        projections = rel_verts @ normals.T # (6, 6)
        
        # For each normal k, we need max projection
        # val_k = max_j ( (v_j - c) . n_k )
        # We want max over all k and j
        # Actually, we just need the global max over all vertices and all normals
        # Because S is determined by the furthest vertex in any of the 6 directions.
        
        max_proj = np.max(projections)
        if max_proj > max_need:
            max_need = max_proj
            
    return max_need / apothem_unit

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset and safety gap."""
        pts = []
        # Increase spacing to ensure gap > 0
        # Touching dist is 2.0. Use 2.1.
        scale = 1.05 
        row_height = math.sqrt(3) * 1.0 * scale
        col_width = 2.0 * 1.0 * scale
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0 * scale, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward. Increased radius to avoid overlap."""
        pts = []
        # Chord length at R=2.5 was 1.29 (overlap).
        # Need chord > 2.0 (touching) -> R > 2.0 / (2*sin(15)) = 2.0 / 0.517 = 3.86.
        # Use R = 4.0 to be safe.
        R = 4.0
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
        
    def make_cluster():
        """Tight cluster: central hex + ring of 6 + outer 5. Increased radii."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        # R1 touching is sqrt(3) ~ 1.732. Use 1.8.
        R1 = 1.8
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        # R2: Distance from center. Ring 1 is at 1.8.
        # Outer hexagons need to be further.
        # Use R2 = 3.5
        R2 = 3.5
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid_fast(cfg):
            return None
        return _outer_hex_side_fast(cfg, center, angle)
    
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
        # Fallback: spread out far to ensure validity
        cfg = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 10.0
            cfg.append([r * math.cos(theta), r * math.sin(theta), 0.0])
        best_cfg = cfg
        best_score = 20.0 # Approx side length for R=10 ring
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
    
    # Time allocation: SA gets most time
    nm_time_budget = TIME_LIMIT - 10.0  # leave 10s for NM + safety
    
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
    if remaining_time > 5.0:
        try:
            # Simple Nelder-Mead implementation
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
            
            v0 = flatten(best_cfg, best_center, best_angle)
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
            
            max_iters = 2000 # Reduced iters because objective is faster, but still bounded by time
            iter_count = 0
            nm_deadline = time.time() + remaining_time - 2.0
            
            while iter_count < max_iters and time.time() < nm_deadline:
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
            
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([best_v[3*i], best_v[3*i+1], best_v[3*i+2]])
            center_polished = [best_v[3*N], best_v[3*N+1]]
            angle_polished = best_v[3*N+2]
            
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
