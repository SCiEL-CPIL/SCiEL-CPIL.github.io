import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 142.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            gap = min2 - max1
            return gap
        if max2 < min1:
            gap = min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(xys_rots, beta=30.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    centers = xys_rots[:, :2]
    
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.2:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        penalty = smooth_overlap_penalty(xys_rots, beta=30.0)
        
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring12(gap=0.0):
        """12 hexagons in a ring around center."""
        pts = []
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_compact_cluster(gap=0.0):
        """Create a compact 3x4 grid with slight rotation for better packing"""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(3):
            y = r * spacing_y
            for c in range(4):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        angle = np.radians(15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = np.column_stack([
            pts[:, 0] * cos_a - pts[:, 1] * sin_a,
            pts[:, 0] * sin_a + pts[:, 1] * cos_a
        ])
        pts[:, :2] = rotated
        return pts
    
    def make_staggered_rows(gap=0.0):
        """Create staggered rows with alternating offsets"""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(4):
            y = r * spacing_y
            for c in range(3):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def is_valid(params, tol=1e-6):
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True
    
    def get_L(params):
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    best_L = float('inf')
    best_params = None
    
    gap = 0.05
    configs = []
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_compact_cluster(gap=gap))
    configs.append(make_staggered_rows(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:6]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 7.5, 22.5, 37.5, 52.5]
    
    # Phase 1: Quick evaluation of all configs to find top candidates
    candidate_list = []
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 40:
            break
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 40:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            if is_valid(init_params):
                L_val = get_L(init_params)
                candidate_list.append((L_val, init_params.copy()))
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
    
    # Sort candidates by L and focus optimization on top ones
    candidate_list.sort(key=lambda x: x[0])
    top_candidates = candidate_list[:min(8, len(candidate_list))]
    
    # Phase 2: Thorough optimization of top candidates
    for L_val, init_params in top_candidates:
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # L-BFGS-B with more iterations
        try:
            res_lbgfs = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 400,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 1000
                }
            )
            if is_valid(res_lbgfs.x):
                L_new = get_L(res_lbgfs.x)
                if L_new < best_L:
                    best_L = L_new
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() - start_time > DEADLINE - 20:
            continue
        try:
            res_nm = minimize(
                objective,
                best_params if best_params is not None else init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if is_valid(res_nm.x):
                L_new = get_L(res_nm.x)
                if L_new < best_L:
                    best_L = L_new
                    best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() - start_time < DEADLINE - 12:
        try:
            res_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 600,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            if is_valid(res_final.x):
                L_new = get_L(res_final.x)
                if L_new < best_L:
                    best_L = L_new
                    best_params = res_final.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time < DEADLINE - 8:
            try:
                res_polish = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-11,
                        'gtol': 1e-9,
                        'maxfun': 600
                    }
                )
                if is_valid(res_polish.x):
                    L_new = get_L(res_polish.x)
                    if L_new < best_L:
                        best_L = L_new
                        best_params = res_polish.x.copy()
            except Exception:
                pass
        
        # Final micro-polish with very tight tolerance
        if time.time() - start_time < DEADLINE - 5:
            try:
                res_micro = minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if is_valid(res_micro.x):
                    L_new = get_L(res_micro.x)
                    if L_new < best_L:
                        best_L = L_new
                        best_params = res_micro.x.copy()
            except Exception:
                pass
    
    # Fallback: use the best known configuration
    if best_params is None:
        fallback_cfg = make_hex_grid(3, 4, gap=0.1)
        inner = [[float(x), float(y), float(r)] for x, y, r in fallback_cfg]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
