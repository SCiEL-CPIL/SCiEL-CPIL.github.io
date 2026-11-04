import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def smooth_max(values, tau=0.01):
    """Numerically stable soft max using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m == 0:
        return 0.0
    return tau * (m + np.log(np.sum(np.exp((values - m) / tau))))

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    N_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)

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
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def sat_overlap_batch(vertices_list):
    """
    Vectorized SAT overlap check for all pairs.
    Returns (has_overlap, penalty) where penalty is sum of squared penetration depths.
    """
    N_pts = len(vertices_list)
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
    
    for i in range(N_pts):
        for j in range(i+1, N_pts):
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

def outer_hex_side_smooth(points, outer_rot_deg, tau=0.01):
    """
    Smooth version of outer hex side length using soft max.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    abs_proj = np.abs(proj)
    sm = smooth_max(abs_proj, tau)
    return (2.0 / np.sqrt(3.0)) * sm

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params, tau=0.01):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        cx = xys_rots[:, 0]
        cy = xys_rots[:, 1]
        
        # 1. Smooth Overlap Penalty using Inradius Circles
        cx_diff = cx[:, None] - cx[None, :]
        cy_diff = cy[:, None] - cy[None, :]
        dist_sq = cx_diff**2 + cy_diff**2
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_sq_pairs = dist_sq[i_idx, j_idx]
        d_pairs = np.sqrt(d_sq_pairs)
        violations = np.maximum(0.0, np.sqrt(3.0) - d_pairs)
        penalty = np.sum(violations**2)
        
        # 2. Enclosing Hexagon Side Length with smooth max
        vertices_list = get_hex_vertices_batch(xys_rots)
        all_points = np.vstack(vertices_list)
        
        L = outer_hex_side_smooth(all_points, outer_rot, tau)
        
        return L + 100.0 * penalty
    
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

    def make_compact_ring(gap=0.0):
        """Alternative compact arrangement: 2 rows of 3 + 3 rows of 2."""
        pts = []
        sqrt3 = np.sqrt(3)
        sx = sqrt3 + gap
        sy = 1.5 + gap * 0.3
        for r in range(4):
            y = r * sy
            for c in range(4):
                x = c * sx
                if r % 2 == 1:
                    x += sx / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts[:N])
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_circular(gap=0.0):
        """12 hexes in a circular pattern."""
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0, 0, 0.0])
        d = sqrt3 + gap
        for k in range(11):
            ang = 2 * np.pi * k / 11.0
            pts.append([d * np.cos(ang), d * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_two_rings(gap=0.0):
        """2 centers + 10 in ring."""
        pts = []
        sqrt3 = np.sqrt(3)
        d_center = sqrt3 / 2 + gap
        d_ring = sqrt3 * 1.5 + gap
        pts.append([-d_center, 0, 0.0])
        pts.append([d_center, 0, 0.0])
        for k in range(10):
            ang = 2 * np.pi * k / 10.0
            pts.append([d_ring * np.cos(ang), d_ring * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations with a small gap to ensure validity
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_compact_ring(gap=gap))
    configs.append(make_circular(gap=gap))
    configs.append(make_two_rings(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 7.5, 22.5]
    
    # Phase 1: Coarse search with Nelder-Mead
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 25:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 25:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            xys_rots_init = init_params[:3*N].reshape(N, 3)
            outer_rot_init = init_params[3*N]
            vertices_list_init = get_hex_vertices_batch(xys_rots_init)
            
            has_overlap, _ = sat_overlap_batch(vertices_list_init)
            
            if not has_overlap:
                L_val = outer_hex_side_batch(vertices_list_init, outer_rot_init)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            try:
                # Use Nelder-Mead with adaptive options
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 400,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                # Evaluate the result
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
    
    # Phase 2: Fine-tune with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
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
    if best_params is not None and time.time() - start_time < DEADLINE - 8:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
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
    
    # Fallback: Use the best valid initial config or a simple valid grid
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Final verification and potential improvement: try small perturbations
    if time.time() - start_time < DEADLINE - 5:
        try:
            rng = np.random.default_rng(42)
            for trial in range(30):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                perturbation = rng.normal(0, 0.005, size=best_params.shape)
                perturbation[3*N] = 0
                perturbed = best_params + perturbation
                
                xys_rots_p = perturbed[:3*N].reshape(N, 3)
                outer_rot_p = perturbed[3*N]
                
                vertices_list_p = get_hex_vertices_batch(xys_rots_p)
                has_overlap_p, _ = sat_overlap_batch(vertices_list_p)
                
                if not has_overlap_p:
                    L_val_p = outer_hex_side_batch(vertices_list_p, outer_rot_p)
                    if L_val_p < best_L:
                        best_L = L_val_p
                        best_params = perturbed.copy()
                        
        except Exception:
            pass
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
