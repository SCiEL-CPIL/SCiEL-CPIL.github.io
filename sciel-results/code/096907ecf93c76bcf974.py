import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

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
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def smooth_hex_distance(centers_i, centers_j, rot_i, rot_j):
    """
    Smooth approximation of the minimum distance between two unit hexagons.
    Uses the Minkowski sum idea: distance between two convex polygons >=
    distance between their centers minus the sum of their support functions
    in the direction of the center-to-center vector.
    
    For regular hexagons, the support function in direction theta is:
    h(theta) = (sqrt(3)/2) / cos((theta - nearest_60deg) mod 60deg - 30deg)
    but capped. We use a smooth version.
    
    Returns a smooth lower bound on the actual distance (negative = likely overlapping).
    """
    d_vec = centers_j - centers_i
    d = np.linalg.norm(d_vec)
    if d < 1e-10:
        return -2.0  # definitely overlapping
    
    d_dir = d_vec / d
    
    # For a regular hexagon with side 1, the inradius (apothem) is sqrt(3)/2
    # The support function h(theta) gives the max projection in direction theta
    # For a flat-top hexagon (rot=0), edges are at angles 30, 90, 150, 210, 270, 330
    # h(theta) = apothem / cos(delta) where delta is angle from nearest edge normal
    
    def support_hex(theta_deg, rot_deg):
        """Support function of unit hexagon with given rotation."""
        apothem = np.sqrt(3.0) / 2.0
        # Edge normals are at rot + 30, rot + 90, ... (6 directions)
        edge_normals = np.radians(np.array([30, 90, 150, 210, 270, 330]) + rot_deg)
        theta_rad = np.radians(theta_deg)
        
        # Find closest edge normal
        diffs = np.abs(np.angle(np.exp(1j * (theta_rad - edge_normals))))
        min_diff = np.min(diffs)
        
        # Support = apothem / cos(min_diff) for |min_diff| < 30 deg
        # Beyond that, it's vertex-dominated, but for smoothness we use:
        h = apothem / np.cos(np.clip(min_diff, 0, np.pi/6 - 1e-9))
        return h
    
    theta_deg = math.degrees(math.atan2(d_dir[1], d_dir[0]))
    
    h_i = support_hex(theta_deg, rot_i)
    h_j = support_hex(theta_deg + 180, rot_j)
    
    # Smooth distance: d - h_i - h_j (negative means centers too close)
    smooth_dist = d - h_i - h_j
    return smooth_dist

def smooth_overlap_penalty(params, beta=30.0):
    """
    Compute smooth overlap penalty.
    Uses both SAT-based exact check and smooth Minkowski-based distance.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    rots = xys_rots[:, 2]
    
    # Precompute vertices for all hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    # Compute pairwise penetrations using SAT for accuracy
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 3.5:  # Only compute SAT if close enough
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

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params, penalty_weight=100.0):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        penalty = smooth_overlap_penalty(params, beta=30.0)
        
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + penalty_weight * penalty
    
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
        """1 center + 6 ring + 5 outer in a hexagonal cluster."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        # 5 outer positions (one missing to make 12 total)
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
    
    def make_staggered(gap=0.0):
        """Staggered 2x6 arrangement with alternating rotations."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(2):
            y = r * spacing_y
            for c in range(6):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                rot = 0.0 if r == 0 else 30.0
                pts.append([x, y, rot])
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
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_staggered(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:5]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    # Add some perturbed variants for diversity
    for base_cfg in configs[:3]:
        perturbed = base_cfg.copy()
        perturbed[:, 0] += np.random.normal(0, 0.1, N)
        perturbed[:, 1] += np.random.normal(0, 0.1, N)
        configs.append(perturbed)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 60.0]
    
    # Phase 1: Evaluate initial configurations and run optimization
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 20:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check initial validity
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # L-BFGS-B optimization with annealed penalty
            try:
                # First pass: high penalty weight
                res1 = minimize(
                    lambda p: objective(p, penalty_weight=200.0),
                    init_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 400
                    }
                )
                
                if is_valid(res1.x):
                    L_val = get_L(res1.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res1.x.copy()
                
                # Second pass: lower penalty weight for better L
                if time.time() - start_time < DEADLINE - 25:
                    start2 = res1.x if is_valid(res1.x) else init_params
                    res2 = minimize(
                        lambda p: objective(p, penalty_weight=50.0),
                        start2,
                        method='L-BFGS-B',
                        options={
                            'maxiter': 150,
                            'ftol': 1e-9,
                            'gtol': 1e-7,
                            'maxfun': 400
                        }
                    )
                    
                    if is_valid(res2.x):
                        L_val = get_L(res2.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res2.x.copy()
                            
            except Exception:
                pass
            
            if time.time() - start_time > DEADLINE - 20:
                continue
                
            # Nelder-Mead polish
            try:
                polish_start = best_params if best_params is not None else init_params
                
                res_nm = minimize(
                    lambda p: objective(p, penalty_weight=100.0),
                    polish_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                if is_valid(res_nm.x):
                    L_val = get_L(res_nm.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
                        
            except Exception:
                pass
    
    # Final high-precision polish passes
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        # Pass 1: Nelder-Mead with very tight tolerance
        try:
            res_final = minimize(
                lambda p: objective(p, penalty_weight=500.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # Pass 2: L-BFGS-B with very tight tolerance
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        try:
            res_final2 = minimize(
                lambda p: objective(p, penalty_weight=1000.0),
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 1000
                }
            )
            
            if is_valid(res_final2.x):
                L_val = get_L(res_final2.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final2.x.copy()
                    
        except Exception:
            pass
    
    # Pass 3: Final Nelder-Mead
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res_final3 = minimize(
                lambda p: objective(p, penalty_weight=500.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final3.x):
                L_val = get_L(res_final3.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final3.x.copy()
                    
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
    
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
