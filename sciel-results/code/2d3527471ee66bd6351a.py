"""Improved program for hexagon packing; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Outward normals at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp (vectorized).
    inner: list of [x, y, angle_deg]
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    n = len(inner)
    
    # Precompute normals in outer frame
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)  # (6,2)
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # All 6 vertices of unit hexagon at origin, angle 0
    base_angles = np.arange(6) * np.pi / 3.0
    base_verts = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)  # (6,2)
    
    # For each inner hexagon: rotate base verts by its angle, translate
    # inner angles in radians
    iangles = np.radians(np.array([inner[i][2] for i in range(n)]))
    
    # cos/sin of each inner angle
    ci = np.cos(iangles)
    si = np.sin(iangles)
    
    # Rotate base vertices: R(iangle) @ base_verts
    # For each hex i, rotated_verts = [[ci_i, -si_i], [si_i, ci_i]] @ base_verts
    # = [ci_i * bx - si_i * by, si_i * bx + ci_i * by] for each base vertex
    # Shape: (n, 6, 2)
    bx = base_verts[:, 0]  # (6,)
    by = base_verts[:, 1]  # (6,)
    
    rx = ci[:, None] * bx[None, :] - si[:, None] * by[None, :]  # (n, 6)
    ry = si[:, None] * bx[None, :] + ci[:, None] * by[None, :]  # (n, 6)
    
    # Translate by center
    ix_arr = np.array([inner[i][0] for i in range(n)])
    iy_arr = np.array([inner[i][1] for i in range(n)])
    
    vx = rx + ix_arr[:, None]  # (n, 6)
    vy = ry + iy_arr[:, None]  # (n, 6)
    
    # Flatten to (n*6, 2)
    vx_flat = vx.ravel()  # (36,)
    vy_flat = vy.ravel()  # (36,)
    
    # Rotate into outer frame
    dx = vx_flat - ox
    dy = vy_flat - oy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # Project onto 6 normals: (36, 6)
    # proj[i, j] = lx[i]*nx[j] + ly[i]*ny[j]
    nx_arr = normals[:, 0]  # (6,)
    ny_arr = normals[:, 1]  # (6,)
    projections = lx[:, None] * nx_arr[None, :] + ly[:, None] * ny_arr[None, :]  # (36, 6)
    
    all_projs = projections.ravel()  # (216,)
    
    max_proj = np.max(all_projs)
    if max_proj < -1000:
        return 0.0
    
    exp_sum = np.sum(np.exp(beta * (all_projs - max_proj)))
    soft_max = max_proj + np.log(exp_sum) / beta
    
    return float(2.0 * soft_max / SQRT3)

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.01):
    """
    Compute a smooth penalty for overlaps based on center distances.
    Uses a tighter threshold since hexagons with same orientation need dist >= 2*apothem = SQRT3.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Define 11 hexagons on a compact hexagonal lattice
    # Use a 3-4-3-1 or optimized polyhex cluster for minimal bounding hexagon
    # Based on exhaustive polyhex search: use a compact arrangement
    # Axial coordinates for a tight 11-hex cluster (3-4-3-1 pattern)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # center + 6 ring
        (2, -1), (1, 1), (-1, 2), (-2, 1)  # 4 more for 11 total
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # Convert axial to Cartesian with tight spacing
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate centroid
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # 3. Optimize Outer Hexagon parameters (cx, cy, angle)
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    best_result = None
    best_val = float('inf')
    
    # Try diverse initial angles
    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 40:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 8000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint Optimization with smooth surrogate (vectorized, faster)
    beta = 80.0
    
    def objective_joint_smooth(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Local search with decreasing step sizes - more iterations due to faster eval
    step_size = 0.05
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline - 25:
            break
        
        num_perturb = 4
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            if idx % 3 == 2:
                delta = np.random.normal(0, step_size * 0.5)
            else:
                delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
            
        neighbor_val = objective_joint_smooth(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            if np.random.random() < 0.05:
                current_params = neighbor
                current_val = neighbor_val
        
        step_size *= 0.997
    
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        best_inner = initial_inner
        best_side = best_val

    # 5. Final Polish with more diverse starting points and tighter tolerance
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Generate diverse polish starting angles
    base_angle = best_outer_angle
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1], base_angle + 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle + 5.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 5.0],
        [best_outer_center[0] + 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, base_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, base_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (float(best_polish_params[0]), float(best_polish_params[1]))
        best_outer_angle = float(best_polish_params[2])
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
