import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    n_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)  # (N, 6, 2)

def sat_overlap_batch(vertices_list):
    """
    Vectorized SAT overlap check for all pairs.
    Returns (has_overlap, penalty) where penalty is sum of squared penetration depths.
    """
    n_pts = len(vertices_list)
    axes_all = []
    for v in vertices_list:
        hex_axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                hex_axes.append(n)
        axes_all.append(hex_axes)
    
    has_overlap = False
    penalty = 0.0
    
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            all_axes = axes_all[i] + axes_all[j]
            
            min_overlap = float('inf')
            separated = False
            for n in all_axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                
                overlap = min(max1, max2) - max(min1, min2)
                min_overlap = min(min_overlap, overlap)
            
            if not separated:
                if min_overlap > 1e-9:
                    has_overlap = True
                    penalty += min_overlap ** 2
    
    return has_overlap, penalty

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_soft(points, outer_rot_deg, beta=50.0):
    """
    Smooth surrogate for outer hex side using soft-max.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    abs_proj = np.abs(proj)
    
    # Soft-max: log-sum-exp approximation of max
    scaled = beta * abs_proj
    max_scaled = np.max(scaled)
    soft_max = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return (2.0 / np.sqrt(3.0)) * soft_max

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def make_hexagonal_lattice_config(spacing, rot_offset=0.0):
    """
    Generate a hexagonal lattice of 12 unit hexagons.
    spacing controls the distance between centers (must be >= sqrt(3) for non-overlap).
    Returns (N, 3) array of [x, y, angle_degrees].
    """
    pts = []
    sqrt3 = np.sqrt(3)
    # Use a hexagonal arrangement: rows with alternating offsets
    # For 12 hexagons, use 3 rows of 4, or 4 rows of 3, etc.
    # Let's use a compact arrangement: 2-3-4-3 pattern or similar
    
    # Actually, let's use a proper hexagonal lattice.
    # A hexagonal lattice of 12 points can be arranged as:
    # Row 0: 4 points
    # Row 1: 3 points (offset)
    # Row 2: 4 points
    # Row 3: 1 point
    # Or more symmetrically:
    # Center + 6 around + 5 more
    
    # Let's use the "flower" pattern: 1 center, 6 around, 5 more in a ring
    d1 = spacing
    d2 = 2 * spacing
    
    pts.append([0, 0, rot_offset])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), rot_offset])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), rot_offset])
    
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring_config(spacing, rot_offset=0.0):
    """Two concentric rings: 6 inner + 6 outer."""
    pts = []
    d1 = spacing
    d2 = 2 * spacing
    
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), rot_offset])
    for k in range(6):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), rot_offset])
    
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_rect_grid_config(spacing, rot_offset=0.0):
    """Rectangular grid 4x3."""
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 * spacing
    dy = 1.5 * spacing
    
    for r in range(3):
        for c in range(4):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += dx / 2.0
            pts.append([x, y, rot_offset])
    
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective_smooth(params, beta=30.0):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        has_overlap, penalty = sat_overlap_batch(vertices_list)
        
        all_points = np.vstack(vertices_list)
        L_soft = outer_hex_side_soft(all_points, outer_rot, beta)
        
        return L_soft + 20.0 * penalty
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        vertices_list = get_hex_vertices_batch(xys_rots)
        
        has_overlap, penalty = sat_overlap_batch(vertices_list)
        
        L = outer_hex_side_batch(vertices_list, outer_rot)
        
        return L + 50.0 * penalty
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice with various spacings
    for spacing in [sqrt3 + 0.01, sqrt3 + 0.05, sqrt3 + 0.1, sqrt3 + 0.2]:
        configs.append(make_hexagonal_lattice_config(spacing, 0.0))
        configs.append(make_hexagonal_lattice_config(spacing, 30.0))
    
    # Ring configurations
    for spacing in [sqrt3 + 0.01, sqrt3 + 0.05, sqrt3 + 0.1]:
        configs.append(make_ring_config(spacing, 0.0))
        configs.append(make_ring_config(spacing, 30.0))
    
    # Rectangular grid
    for spacing in [1.0, 1.05, 1.1]:
        configs.append(make_rect_grid_config(spacing, 0.0))
        configs.append(make_rect_grid_config(spacing, 30.0))
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    from scipy.optimize import minimize
    
    # Phase 1: Quick search with smooth objective
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        for orot in outer_rots[:2]:
            if time.time() - start_time > DEADLINE - 40:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            try:
                res = minimize(
                    objective_smooth, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices_list = get_hex_vertices_batch(xys_rots)
                
                has_overlap, _ = sat_overlap_batch(vertices_list)
                
                if not has_overlap:
                    L_val = outer_hex_side_batch(vertices_list, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Phase 2: Refine best solution with exact objective
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            
            has_overlap, _ = sat_overlap_batch(vertices_list)
            
            if not has_overlap:
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_params is not None and time.time() - start_time < DEADLINE - 12:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            
            has_overlap, _ = sat_overlap_batch(vertices_list)
            
            if not has_overlap:
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
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
