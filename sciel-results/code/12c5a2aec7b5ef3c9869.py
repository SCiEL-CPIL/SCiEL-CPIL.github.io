import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize

N = 12
DEADLINE = 140.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

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

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    """
    Find the optimal rotation of the outer hexagon to minimize side length.
    """
    best_L = float('inf')
    best_rot = init_rot
    
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    # Local refinement
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    return best_L, best_rot

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
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    """Create a hex grid of unit hexagons with optional gap."""
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
    """1 center + 6 ring + 5 outer with gap."""
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

def is_valid(params, tol=1e-6):
    """Check if configuration has no overlaps."""
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def residuals_least_squares(params):
    """
    Residuals for least_squares:
    - For each pair of inner hexagons: max(0, -sat_overlap) (penetration depth)
    - For each inner hexagon vertex: max(0, distance_to_outer_boundary - margin)
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    # Precompute vertices for all hexagons
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    # Pairwise overlap residuals
    pair_residuals = []
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            # If d < 0, they overlap; residual is -d (positive penetration)
            # If d >= 0, no overlap; residual is 0
            if d < 0:
                pair_residuals.append(-d)
            else:
                pair_residuals.append(0.0)
    
    # Outer boundary residuals: vertices should be inside the outer hexagon
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    max_proj = np.max(np.abs(proj))
    L = (2.0 / np.sqrt(3.0)) * max_proj
    
    # We want to minimize L, so we add residuals that push vertices inward
    # Actually, for least_squares, we want residuals to be small.
    # The outer constraint is that all points must be inside the hexagon.
    # Since we're optimizing L as a separate objective, let's just return pair residuals
    # and handle L separately.
    
    return np.array(pair_residuals)

def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m < -50.0 / beta:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def objective_combined(params, beta=50.0):
    """
    Combined objective: smooth outer side length + smooth penalty for overlaps.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    # Compute smooth overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < 0:
                penalty += d * d
    
    # Compute smooth outer side length using smooth max
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    abs_proj = np.abs(proj)
    # smooth_max over all projections
    max_proj_smooth = smooth_max(abs_proj.flatten(), beta=beta)
    L_smooth = (2.0 / np.sqrt(3.0)) * max_proj_smooth
    
    return L_smooth + 100.0 * penalty

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0]
    
    for cfg in configs:
        if time_left() < 15:
            break
        
        for orot in outer_rots:
            if time_left() < 15:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                xys_rots = init_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Use least_squares to optimize
            if time_left() < 20:
                continue
            
            try:
                def residuals(params):
                    xys_rots = params[:3*N].reshape(N, 3)
                    outer_rot = params[3*N]
                    
                    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
                    
                    # Pairwise overlap residuals
                    res = []
                    for i in range(N):
                        for j in range(i+1, N):
                            d = sat_overlap(vertices_list[i], vertices_list[j])
                            if d < 0:
                                res.append(-d)
                            else:
                                res.append(0.0)
                    
                    # Outer boundary: we want to minimize the side length
                    # Add a residual that is proportional to how much each vertex
                    # exceeds the boundary of a hexagon with side length L_target
                    # This is tricky because L depends on all points.
                    # Instead, let's just use the pair residuals and handle L separately.
                    
                    return np.array(res)
                
                res_ls = least_squares(
                    residuals,
                    init_params,
                    method='trf',
                    max_nfev=100,
                    ftol=1e-6,
                    gtol=1e-6
                )
                
                if is_valid(res_ls.x):
                    xys_rots = res_ls.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_ls.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_ls.x.copy()
                        
            except Exception:
                pass
            
            # Nelder-Mead polish
            if time_left() < 25:
                continue
            
            try:
                polish_start = best_params if best_params is not None else init_params
                
                from scipy.optimize import minimize
                res_nm = minimize(
                    objective_combined,
                    polish_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                if is_valid(res_nm.x):
                    xys_rots = res_nm.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_nm.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
                        
            except Exception:
                pass
    
    # Optimize outer rotation independently
    if best_params is not None and time_left() > 25:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Final high-precision polish with multiple passes
    if best_params is not None and time_left() > 15:
        try:
            # Pass 1: Nelder-Mead with smooth objective
            res_final = minimize(
                lambda p: objective_combined(p, beta=30.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_final.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
        
        # Pass 2: L-BFGS-B with smooth objective
        if best_params is not None and time_left() > 10:
            try:
                res_lbfgs = minimize(
                    lambda p: objective_combined(p, beta=20.0),
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-8
                    }
                )
                
                if is_valid(res_lbfgs.x):
                    xys_rots = res_lbfgs.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_lbfgs.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_lbfgs.x.copy()
                        
            except Exception:
                pass
        
        # Pass 3: Fine Nelder-Mead polish
        if best_params is not None and time_left() > 5:
            try:
                res_polish = minimize(
                    lambda p: objective_combined(p, beta=100.0),
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-9,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
                
                if is_valid(res_polish.x):
                    xys_rots = res_polish.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res_polish.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_polish.x.copy()
                        
            except Exception:
                pass
    
    # One more outer rotation optimization
    if best_params is not None and time_left() > 5:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Final micro-polish of outer rotation
    if best_params is not None and time_left() > 2:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            # Very fine search around current best rotation
            cur_rot = best_params[3*N]
            fine_rots = np.linspace(cur_rot - 0.1, cur_rot + 0.1, 200)
            best_local_L = best_L
            best_local_rot = cur_rot
            for rot in fine_rots:
                L = outer_hex_side(all_points, rot)
                if L < best_local_L:
                    best_local_L = L
                    best_local_rot = rot
            
            if best_local_L < best_L:
                best_L = best_local_L
                best_params[3*N] = best_local_rot
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
