import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def compute_sat_penetration(v1, v2):
    """
    Computes the minimum penetration depth (positive if overlapping, 0 if separated).
    Returns the overlap depth.
    """
    # v1, v2 are (6, 2) arrays
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    # Stack axes into (12, 2)
    axes = np.array(axes)
    
    # Project v1 and v2 onto all axes
    # proj1: (12, 6), proj2: (12, 6)
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)
    
    # Overlap on each axis
    # If max1 < min2 or max2 < min1, there is a gap (separation)
    gap1 = min2 - max1
    gap2 = min1 - max2
    
    # If gap > 0, they are separated on this axis.
    # We need to check if ANY axis has a positive gap.
    # If all gaps are negative, they overlap.
    # The penetration depth is the minimum overlap (most negative gap) among axes with no gap?
    # Actually, the penetration depth is the minimum amount one must move to separate.
    # This is -max(min_overlap).
    # Wait, if they are separated on one axis, penetration is 0.
    
    # Let's find the maximum gap. If max_gap > 0, they are separated.
    # If max_gap < 0, they are overlapping. The depth is -max_gap? No.
    # If they overlap on all axes, the minimum separation required is the minimum overlap.
    # Overlap on axis k is min(max1, max2) - max(min1, min2).
    # If this is positive, they overlap on this axis.
    # The total overlap depth is the minimum of these overlaps.
    
    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
    
    # If any overlap is <= 0, they are separated (or touching).
    # Actually, if overlaps[k] <= 0, it means they are separated on axis k.
    # So if np.max(overlaps) <= 0, they are separated. Penetration = 0.
    # If np.max(overlaps) > 0, they overlap on all axes.
    # The penetration depth is np.min(overlaps).
    
    if np.max(overlaps) <= 0:
        return 0.0
    else:
        return np.min(overlaps)

def smooth_overlap_penalty(params, beta=20.0):
    """
    Compute smooth overlap penalty.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5: # Rough bounding box check
                pen = compute_sat_penetration(vertices_list[i], vertices_list[j])
                if pen > 0:
                    # Smooth max(0, pen)
                    # Using softplus: log(1 + exp(beta * pen)) / beta
                    # But we want to penalize overlap, so we want to push pen to 0.
                    # The gradient of softplus is sigmoid(beta * pen).
                    # This is smooth.
                    total_penalty += (1.0 / beta) * np.log(1.0 + np.exp(beta * pen))
    
    return total_penalty

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
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        penalty = smooth_overlap_penalty(params, beta=20.0)
        
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 50.0 * penalty
    
    def is_valid(params, tol=1e-6):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                pen = compute_sat_penetration(vertices_list[i], vertices_list[j])
                if pen > tol:
                    return False
        return True

    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
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
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    gap = 0.01
    configs = []
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time_left() < 10:
            break
        
        for orot in outer_rots:
            if time_left() < 5:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check initial validity
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Optimization
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
                )
                
                if is_valid(res.x):
                    L_val = get_L(res.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
            
            # Polish
            if best_params is not None and time_left() > 15:
                try:
                    res_polish = minimize(
                        objective,
                        best_params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
                    )
                    if is_valid(res_polish.x):
                        L_val = get_L(res_polish.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res_polish.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_L, outer_rot
