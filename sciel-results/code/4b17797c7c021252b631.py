"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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

def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at origin with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Transform point into hexagon's local frame
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0
    
    # Check if point is inside hexagon
    inside = True
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > apothem + 1e-12:
            inside = False
            break
        if proj > max_proj:
            max_proj = proj
    
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

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Approximate SDF between two hexagons using vertex sampling."""
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    min_d = float('inf')
    # For each vertex of hex1, compute distance to hex2
    for (vx, vy) in verts1:
        d = _sdf_hex_point(vx - cx2, vy - cy2, a2, side)
        if d < min_d:
            min_d = d
    # For each vertex of hex2, compute distance to hex1
    for (vx, vy) in verts2:
        d = _sdf_hex_point(vx - cx1, vy - cy1, a1, side)
        if d < min_d:
            min_d = d
    return min_d

def _smooth_objective(v, sigma=0.15):
    """Smooth objective: outer side length + collision penalty using SDF."""
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    s = _outer_hex_side(cfg, center, angle)
    
    # Collision penalty using SDF
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sdf_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2],
                            cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 2.0 * sigma:
                penalty += math.exp(-d / sigma) - 1.0
    
    return s + 10.0 * penalty

def _smooth_gradient(v, sigma=0.15):
    """Numerical gradient of smooth objective."""
    eps = 1e-5
    grad = np.zeros_like(v)
    f0 = _smooth_objective(v, sigma)
    for i in range(len(v)):
        v_pert = v.copy()
        v_pert[i] += eps
        f_pert = _smooth_objective(v_pert, sigma)
        grad[i] = (f_pert - f0) / eps
    return grad

def _lbfgs_polish(cfg_init, center_init, angle_init, time_budget):
    """L-BFGS-B optimization for polishing."""
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return np.array(v)
    
    v0 = flatten(cfg_init, center_init, angle_init)
    n_params = len(v0)
    
    best_v = v0.copy()
    best_f = _smooth_objective(v0)
    
    # Multiple L-BFGS-B runs with different initial perturbations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - t0 > time_budget:
            break
        
        if restart == 0:
            v_start = v0.copy()
        else:
            v_start = v0 + np.random.normal(0, 0.1, n_params)
        
        try:
            result = minimize(
                _smooth_objective,
                v_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            f_res = result.fun
            if f_res < best_f:
                best_f = f_res
                best_v = result.x
        except Exception:
            pass
    
    return best_v, best_f

def _nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
    """Nelder-Mead simplex optimization for polishing."""
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return np.array(v)
    
    v0 = flatten(cfg_init, center_init, angle_init)
    n_params = len(v0)
    
    # Initialize simplex
    simplex = [v0.copy()]
    f_vals = [_smooth_objective(v0)]
    
    for i in range(n_params):
        v_pert = v0.copy()
        step = 0.05 if i % 3 != 2 else 0.02
        v_pert[i] += step
        simplex.append(v_pert)
        f_vals.append(_smooth_objective(v_pert))
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_idx = min(range(n_params + 1), key=lambda i: f_vals[i])
    best_f = f_vals[best_idx]
    best_v = simplex[best_idx].copy()
    
    max_iters = 2000
    iter_count = 0
    no_improve_count = 0
    
    while iter_count < max_iters and time.time() - t0 < time_budget:
        # Sort simplex by objective value
        order = sorted(range(n_params + 1), key=lambda i: f_vals[i])
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if f_vals[-1] - f_vals[0] < 1e-10:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        worst_v = simplex[-1]
        v_reflect = centroid + alpha * (centroid - worst_v)
        f_reflect = _smooth_objective(v_reflect)
        
        if f_reflect < f_vals[0]:
            v_expand = centroid + gamma * (v_reflect - centroid)
            f_expand = _smooth_objective(v_expand)
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
            v_contract = centroid + rho * (worst_v - centroid)
            f_contract = _smooth_objective(v_contract)
            if f_contract < f_vals[-1]:
                simplex[-1] = v_contract
                f_vals[-1] = f_contract
            else:
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = _smooth_objective(simplex[i])
        
        improved = False
        for i in range(n_params + 1):
            if f_vals[i] < best_f:
                best_f = f_vals[i]
                best_v = simplex[i].copy()
                improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 80:
                break
        
        iter_count += 1
    
    return best_v, best_f

def _anneal_polish(cfg_init, center_init, angle_init, time_budget):
    """Simulated annealing polish."""
    cfg = [row[:] for row in cfg_init]
    center = list(center_init)
    angle = angle_init
    
    cur_score = None
    if _is_valid(cfg):
        cur_score = _outer_hex_side(cfg, tuple(center), angle)
    
    best_score = cur_score
    best_cfg = [row[:] for row in cfg]
    best_center = center[:]
    best_angle = angle
    
    T = 0.3
    T_end = 0.001
    alpha = 0.999
    
    while time.time() - t0 < time_budget:
        r = random.random()
        
        if r < 0.5:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.3 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.7:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 3.0 + 2.0 * T)
        elif r < 0.8:
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (1.0 + 0.3 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.9:
            old_angle = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.12 * (1.0 + 0.3 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        new_score = None
        if _is_valid(cfg):
            new_score = _outer_hex_side(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (cur_score is None) or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.5:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.7:
                    cfg[idx][2] = old_ang
                elif r < 0.8:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.9:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            if r < 0.5:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.7:
                cfg[idx][2] = old_ang
            elif r < 0.8:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.9:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha)
    
    return best_cfg, best_center, best_angle, best_score

def construct():
    global t0
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    def make_hex_lattice():
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
    
    def make_ring():
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
    
    def make_center_ring():
        """1 center + 6 ring + 5 outer."""
        pts = []
        sqrt3 = math.sqrt(3)
        d1 = sqrt3 + 0.05
        d2 = 2 * sqrt3 + 0.05
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), math.degrees(ang)])
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), math.degrees(ang)])
        return pts
    
    def make_grid_initial():
        """Grid-like arrangement with rotation variety."""
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
    
    inits = [make_hex_lattice(), make_ring(), make_center_ring(), make_grid_initial()]
    outer_angles = [0.0, 15.0, 30.0, -15.0, 45.0]
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
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
    
    # Phase 1: L-BFGS-B polish with multiple restarts
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 30.0:
        lbfgs_deadline = time.time() + min(25.0, remaining * 0.4)
        try:
            v_polished, f_polished = _lbfgs_polish(best_cfg, best_center, best_angle, lbfgs_deadline)
            cfg_p = []
            for i in range(N):
                cfg_p.append([v_polished[3*i], v_polished[3*i+1], v_polished[3*i+2]])
            center_p = [v_polished[3*N], v_polished[3*N+1]]
            angle_p = v_polished[3*N+2]
            
            score_p = full_score(cfg_p, tuple(center_p), angle_p)
            if score_p is not None and score_p < best_score:
                best_score = score_p
                best_cfg = cfg_p
                best_center = center_p
                best_angle = angle_p
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 20.0:
        nm_deadline = time.time() + min(15.0, remaining * 0.4)
        try:
            v_nm, f_nm = _nelder_mead_polish(best_cfg, best_center, best_angle, nm_deadline)
            cfg_nm = []
            for i in range(N):
                cfg_nm.append([v_nm[3*i], v_nm[3*i+1], v_nm[3*i+2]])
            center_nm = [v_nm[3*N], v_nm[3*N+1]]
            angle_nm = v_nm[3*N+2]
            
            score_nm = full_score(cfg_nm, tuple(center_nm), angle_nm)
            if score_nm is not None and score_nm < best_score:
                best_score = score_nm
                best_cfg = cfg_nm
                best_center = center_nm
                best_angle = angle_nm
        except Exception:
            pass
    
    # Phase 3: Simulated annealing polish
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 15.0:
        sa_deadline = time.time() + min(10.0, remaining * 0.5)
        try:
            cfg_sa, center_sa, angle_sa, score_sa = _anneal_polish(
                best_cfg, best_center, best_angle, sa_deadline
            )
            if score_sa is not None and score_sa < best_score:
                best_score = score_sa
                best_cfg = cfg_sa
                best_center = center_sa
                best_angle = angle_sa
        except Exception:
            pass
    
    # Final verification
    final_score = full_score(best_cfg, tuple(best_center), best_angle)
    if final_score is None:
        # Fallback to a simple valid configuration
        cfg = make_hex_lattice()
        final_score = full_score(cfg, (0.0, 0.0), 0.0)
        if final_score is not None:
            best_cfg = cfg
            best_center = [0.0, 0.0]
            best_angle = 0.0
            best_score = final_score
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
