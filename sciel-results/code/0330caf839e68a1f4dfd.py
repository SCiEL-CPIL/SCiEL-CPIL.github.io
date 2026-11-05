"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # safety margin below 150s
EPS = 1e-9

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
        if maxA <= minB + EPS or maxB <= minA + EPS:
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

def _min_overlap_depth(vertsA, vertsB):
    """Compute minimum overlap depth between two convex polygons via SAT."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    min_depth = float('inf')
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= EPS:
            return 0.0
        if overlap < min_depth:
            min_depth = overlap
    return min_depth

def _smooth_collision_penalty(all_verts, beta=8.0):
    """
    Smooth collision penalty using log-sum-exp over overlap depths.
    This creates a differentiable surrogate for the hard collision constraint.
    """
    m = len(all_verts)
    log_sum = 0.0
    max_log = -float('inf')
    
    for i in range(m):
        for j in range(i + 1, m):
            depth = _min_overlap_depth(all_verts[i], all_verts[j])
            if depth > 0:
                # Use exp(-beta * depth) as a smooth penalty that decays with separation
                # Actually, we want penalty when overlapping (depth > 0)
                # Use: penalty = depth^2 is simple but non-smooth at 0
                # Better: use softplus-like: log(1 + exp(beta*depth)) - log(2) for smoothness
                # Or simply: depth^2 works but let's use a smoother version
                # For differentiability, use: 0.5 * depth^2 is fine (smooth)
                # But for the SDF-inspired approach, use: exp(-beta * max(0, -depth)) style
                # Let's use: if overlapping, penalty = depth^2; this is C1 continuous
                log_sum += depth * depth
                if depth * depth > max_log:
                    max_log = depth * depth
    
    return log_sum

def _sdf_based_objective(v, beta=5.0):
    """
    SDF-inspired differentiable objective.
    v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
    
    Uses smooth max for outer hexagon containment and smooth collision penalty.
    The key insight: represent the "distance to constraint violation" smoothly.
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    outer_angle = v[3*N+2]
    
    # Compute all vertices
    all_verts = []
    for p in cfg:
        verts = _hex_vertices(p[0], p[1], p[2])
        all_verts.append(verts)
    
    # Outer hexagon normals
    a = math.radians(outer_angle)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    
    # Smooth max of projections (log-sum-exp for differentiability)
    # We want to minimize the max projection, so we use a smooth approximation
    proj_vals = []
    for verts in all_verts:
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                proj_vals.append(val)
    
    # Soft max using log-sum-exp
    if proj_vals:
        max_val = max(proj_vals)
        if max_val > 1e-10:
            # log-sum-exp: log(sum(exp(beta*(x_i - max_val)))) / beta + max_val
            sum_exp = 0.0
            for val in proj_vals:
                sum_exp += math.exp(beta * (val - max_val))
            smooth_max = max_val + math.log(sum_exp) / beta
        else:
            smooth_max = max_val
    else:
        smooth_max = 0.0
    
    # Smooth collision penalty
    collision_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(all_verts[i], all_verts[j])
            if depth > 0:
                # Quadratic penalty is smooth (C1)
                collision_penalty += depth * depth
    
    return smooth_max + 10.0 * collision_penalty

def construct():
    t0 = time.time()
    random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
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
    
    def make_grid_initial():
        """Grid-like arrangement with some rotation variety."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                ang = random.uniform(-10, 10)
                pts.append([x - 3.0, y - row_height, ang])
        return pts
    
    def make_center_ring_outer():
        """1 center + 6 ring + 5 outer arrangement."""
        pts = []
        sqrt3 = math.sqrt(3)
        d1 = sqrt3 + 0.05
        d2 = 2 * sqrt3 + 0.05
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
        # Center the configuration
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [[p[0] - cx, p[1] - cy, p[2]] for p in pts]
        return pts
    
    def make_compact_cluster():
        """Compact cluster: 3 rows of 4 with tight packing."""
        pts = []
        sqrt3 = math.sqrt(3)
        # Use apothem-based spacing for tight packing
        spacing = sqrt3 + 0.02
        for r in range(3):
            for c in range(4):
                x = c * 2.0 * 1.0 + (1.0 if r % 2 else 0.0)
                y = r * spacing
                pts.append([x - 3.0, y - spacing, 0.0])
        return pts
    
    def nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
        """Nelder-Mead simplex optimization using the smooth SDF-based objective."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return v
        
        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        
        # Initialize simplex
        simplex = [v0[:]]
        f_vals = [_sdf_based_objective(v0)]
        
        for i in range(n_params):
            v_pert = v0[:]
            step = 0.05 if i % 3 != 2 else 0.02
            v_pert[i] += step
            simplex.append(v_pert)
            f_vals.append(_sdf_based_objective(v_pert))
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_idx = min(range(n_params + 1), key=lambda i: f_vals[i])
        best_f = f_vals[best_idx]
        best_v = simplex[best_idx][:]
        
        max_iters = 3000
        iter_count = 0
        no_improve_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            order = sorted(range(n_params + 1), key=lambda i: f_vals[i])
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            if f_vals[-1] - f_vals[0] < 1e-10:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            worst_v = simplex[-1]
            v_reflect = centroid + alpha * (centroid - np.array(worst_v))
            f_reflect = _sdf_based_objective(v_reflect)
            
            if f_reflect < f_vals[0]:
                v_expand = centroid + gamma * (np.array(v_reflect) - centroid)
                f_expand = _sdf_based_objective(v_expand)
                if f_expand < f_reflect:
                    simplex[-1] = v_expand
                    f_vals[-1] = f_expand
                else:
                    simplex[-1] = v_reflect
                    f_vals[-1] = f_reflect
            elif f_reflect < f_vals[-2]:
                simplex[-1] = v_reflect
                f_vals[-1] = f_reflect
            else:
                v_contract = centroid + rho * (np.array(worst_v) - centroid)
                f_contract = _sdf_based_objective(v_contract)
                if f_contract < f_vals[-1]:
                    simplex[-1] = v_contract
                    f_vals[-1] = f_contract
                else:
                    for i in range(1, n_params + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n_params)]
                        f_vals[i] = _sdf_based_objective(simplex[i])
            
            improved = False
            for i in range(n_params + 1):
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_v = simplex[i][:]
                    improved = True
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            
            iter_count += 1
        
        return best_v, best_f
    
    def high_precision_polish(cfg_init, center_init, angle_init, time_budget):
        """
        High-precision coordinate descent with very tight tolerances.
        Uses the exact (non-smooth) objective for final refinement.
        """
        cfg = [row[:] for row in cfg_init]
        center = center_init[:]
        angle = angle_init
        
        def get_score():
            s = full_score(cfg, tuple(center), angle)
            if s is None:
                return float('inf')
            return s
        
        cur_score = get_score()
        best_cfg = [row[:] for row in cfg]
        best_center = center[:]
        best_angle = angle
        best_score = cur_score
        
        n_params = 3 * N + 3
        
        # Very fine step sizes for high-precision polish
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for iteration in range(500):
            if time.time() - t0 > time_budget:
                break
            
            improved = False
            for p in range(n_params):
                if time.time() - t0 > time_budget:
                    break
                
                if p < 3 * N:
                    idx = p // 3
                    param_type = p % 3
                    old_val = cfg[idx][param_type]
                elif p < 3 * N + 2:
                    idx = p - 3 * N
                    old_val = center[idx]
                else:
                    old_val = angle
                
                best_local_val = old_val
                best_local_score = cur_score
                
                for step_size in step_sizes:
                    for direction in [-1, 1]:
                        new_val = old_val + direction * step_size
                        
                        if p < 3 * N:
                            idx = p // 3
                            param_type = p % 3
                            cfg[idx][param_type] = new_val
                        elif p < 3 * N + 2:
                            idx = p - 3 * N
                            center[idx] = new_val
                        else:
                            angle = new_val
                        
                        new_score = get_score()
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_val = new_val
                        
                        # Restore
                        if p < 3 * N:
                            idx = p // 3
                            param_type = p % 3
                            cfg[idx][param_type] = old_val
                        elif p < 3 * N + 2:
                            idx = p - 3 * N
                            center[idx] = old_val
                        else:
                            angle = old_val
                
                if best_local_score < cur_score - 1e-12:
                    if p < 3 * N:
                        idx = p // 3
                        param_type = p % 3
                        cfg[idx][param_type] = best_local_val
                    elif p < 3 * N + 2:
                        idx = p - 3 * N
                        center[idx] = best_local_val
                    else:
                        angle = best_local_val
                    
                    cur_score = best_local_score
                    improved = True
                    
                    if cur_score < best_score:
                        best_score = cur_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
            
            if not improved:
                break
        
        return best_cfg, best_center, best_angle, best_score
    
    # Try multiple initial configurations with diverse outer hexagon params
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_grid_initial(), make_center_ring_outer(), make_compact_cluster()]
    outer_angles = [0.0, 15.0, 30.0, -15.0, 45.0, -30.0]
    for init in inits:
        for oa in outer_angles:
            s = full_score(init, (0.0, 0.0), oa)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = oa
    
    if best_cfg is None:
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
    
    # Simulated annealing phase
    T = 0.8
    T_end = 0.003
    alpha_sa = 0.9997
    
    # Reserve time for polishing
    nm_time_budget = 40.0
    cd_time_budget = 20.0
    sa_time_limit = TIME_LIMIT - nm_time_budget - cd_time_budget
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.25 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.75:
            old_cx, old_cy = center[0], center[1]
            step = 0.15 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            old_angle = angle
            angle += random.gauss(0, 3.0 + 3.0 * T)
        elif r < 0.92:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        else:
            idx1, idx2, idx3 = random.sample(range(N), 3)
            old = [(cfg[i][0], cfg[i][1], cfg[i][2]) for i in (idx1, idx2, idx3)]
            step = 0.15 * (1.0 + 0.5 * T)
            for idx in (idx1, idx2, idx3):
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
        
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
                elif r < 0.92:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    for idx, (ox, oy, oa) in zip((idx1, idx2, idx3), old):
                        cfg[idx][0], cfg[idx][1], cfg[idx][2] = ox, oy, oa
        else:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            elif r < 0.92:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            else:
                for idx, (ox, oy, oa) in zip((idx1, idx2, idx3), old):
                    cfg[idx][0], cfg[idx][1], cfg[idx][2] = ox, oy, oa
        
        T = max(T_end, T * alpha_sa)
    
    # Nelder-Mead polish using smooth SDF-based objective
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            nm_deadline = TIME_LIMIT - cd_time_budget - 5.0
            polished_v, polished_f = nelder_mead_polish(
                best_cfg, best_center, best_angle, nm_deadline
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
    
    # High-precision coordinate descent polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            cd_deadline = TIME_LIMIT - 3.0
            cfg_cd, center_cd, angle_cd, cd_f = high_precision_polish(
                best_cfg, best_center, best_angle, cd_deadline
            )
            cd_score = full_score(cfg_cd, tuple(center_cd), angle_cd)
            if cd_score is not None and cd_score < best_score:
                best_score = cd_score
                best_cfg = cfg_cd
                best_center = center_cd
                best_angle = angle_cd
        except Exception:
            pass
    
    # Final safety check: ensure we have a valid result
    if best_cfg is None or not _is_valid(best_cfg):
        # Fallback to a known-valid configuration
        best_cfg = make_hex_lattice()
        best_center = [0.0, 0.0]
        best_angle = 0.0
        best_score = _outer_hex_side(best_cfg, tuple(best_center), best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
