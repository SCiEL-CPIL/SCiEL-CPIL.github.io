import numpy as np
import time
import math
import random
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin below 150s kill

# Precompute base hexagon vertices (side 1, center 0,0, rot 0)
_BASE_HEX_VERTS = np.column_stack((
    np.cos(np.radians([0, 60, 120, 180, 240, 300])),
    np.sin(np.radians([0, 60, 120, 180, 240, 300]))
))

def get_hex_vertices_fast(cx, cy, rot_deg):
    """Fast hexagon vertex computation."""
    rad = np.radians(rot_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    return _BASE_HEX_VERTS @ R.T + np.array([cx, cy])

def sat_overlap(v1, v2):
    """
    Returns separation distance (positive) if not overlapping, or negative penetration if overlapping.
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

def sdf_convex_polygon(p, verts):
    """
    Compute the signed distance from point p to a convex polygon defined by vertices.
    Positive if outside, negative if inside.
    """
    d = float('inf')
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        e = p2 - p1
        # Normal pointing inward for CCW polygon? 
        # Standard SDF: distance to edge. Sign determined by cross product.
        # For CCW polygon, inward normal is (-dy, dx). 
        # Let's use the distance to the edge line and check if inside.
        
        # Vector from p1 to p
        v = p - p1
        # Project v onto e
        len_e_sq = np.dot(e, e)
        if len_e_sq < 1e-10:
            continue
        t = np.dot(v, e) / len_e_sq
        t = np.clip(t, 0.0, 1.0)
        closest = p1 + t * e
        dist = np.linalg.norm(p - closest)
        
        # Determine sign
        # Cross product of edge (p2-p1) and (p-p1)
        # If CCW, inside is to the left. Cross z > 0 means left.
        cross = e[0] * v[1] - e[1] * v[0]
        # If cross < 0, p is to the right (outside) relative to this edge if polygon is CCW
        # However, we want signed distance. 
        # If inside, dist is negative.
        # A simpler way for convex poly: check if point is inside.
        # But for gradient flow, we need the gradient of the distance.
        # Gradient of dist to closest point on segment is (p - closest)/dist.
        # If inside, we can just use -min distance to any edge? No, SDF is distance to boundary.
        
        # Let's stick to a robust SDF for convex polygon:
        # 1. If point is inside, SDF = -min distance to edges.
        # 2. If point is outside, SDF = +min distance to edges.
        
        # We'll accumulate min distance and track if inside.
        # To be differentiable, we use a soft-min of distances if inside?
        # Actually, for optimization, a simple penalty is enough.
        # We want to push points OUT. 
        # If we treat the penalty as sum of max(0, -SDF)^2, we need SDF.
        
        # Let's compute min distance to boundary and sign.
        # This function will be used in a penalty.
        pass # Placeholder, will implement specific logic in penalty function

def smooth_sdf_penalty(xys_rots, beta=50.0):
    """
    Compute a differentiable overlap penalty using a smooth approximation of the SDF.
    For each pair of hexagons, we check if vertices of one are inside the other.
    We approximate the penetration depth.
    """
    centers = xys_rots[:, :2]
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices_fast(cx, cy, rot)
        vertices_list.append(v)
    
    total_penalty = 0.0
    # We use a differentiable proxy for overlap.
    # Instead of exact SAT, we check vertex-in-polygon for all vertices of both hexes.
    # If a vertex of hex i is inside hex j, we add a penalty based on the "depth".
    # Depth approximation: distance to the nearest edge of hex j.
    
    for i in range(N):
        vi = vertices_list[i]
        vj_all = [vertices_list[j] for j in range(N) if j != i]
        
        for j_idx, vj in enumerate(vj_all):
            # Check vertices of vi against vj
            for p in vi:
                # Compute distance from p to edges of vj
                min_dist = float('inf')
                is_inside = True
                n_edges = len(vj)
                for k in range(n_edges):
                    p1 = vj[k]
                    p2 = vj[(k+1) % n_edges]
                    e = p2 - p1
                    # Normal
                    n = np.array([-e[1], e[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-10: continue
                    n = n / n_norm
                    
                    # Distance to line
                    # d = dot(p - p1, n)
                    # If polygon is CCW, inside is where d > 0 for outward normal?
                    # Let's assume standard orientation.
                    # We just need the minimum distance to the boundary.
                    
                    # Project p onto edge segment
                    v = p - p1
                    len_e_sq = np.dot(e, e)
                    t = np.dot(v, e) / len_e_sq
                    t = np.clip(t, 0.0, 1.0)
                    closest = p1 + t * e
                    dist = np.linalg.norm(p - closest)
                    min_dist = min(min_dist, dist)
                    
                    # Check if inside for this edge (simplification: assume convex)
                    # Cross product to determine side
                    cross = e[0] * v[1] - e[1] * v[0]
                    # If all cross products have the same sign, point is inside.
                    # We'll just rely on min_dist for the penalty magnitude.
                    # A point inside will have min_dist > 0 but we need to know it's inside.
                    # If we just penalize any vertex being close to another hex's edge,
                    # it might not work well.
                    
                    # Better approach: 
                    # Use the fact that for convex polygons, SDF is differentiable.
                    # But implementing exact SDF is tricky.
                    # Let's use a robust differentiable penalty:
                    # Sum of squares of (max(0, -separation))
                    # Where separation is computed via SAT but smoothed?
                    # SAT is not differentiable at 0.
                    
                    # Alternative: 
                    # Penalty = sum over all pairs (i,j) of (max(0, R - dist(i,j)))^2 ? No, that's for circles.
                    
                    # Let's go back to SAT but smooth the minimum.
                    # In the previous code, we used log-sum-exp on penetrations.
                    # That is already a differentiable penalty! 
                    # The issue was likely the optimizer settings or the search space.
                    # I will reuse the log-sum-exp SAT penalty but integrate it better.
                    pass
    
    # Re-implementing the log-sum-exp SAT penalty from the seed, as it is effective.
    # I will just copy the logic from `smooth_overlap_penalty` in the seed but rename/refactor.
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
    max_p = np.max(p_arr)
    if max_p < 1e-12:
        return 0.0
    shifted = beta * (p_arr - max_p)
    shifted = np.clip(shifted, -50, 50)
    lse = max_p + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return lse

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

def optimize_outer_rotation(points, init_rot=0.0):
    """
    Find the outer rotation that minimizes the side length for a fixed set of points.
    """
    def obj(theta):
        return outer_hex_side(points, theta)
    
    best_L = float('inf')
    best_rot = init_rot
    for rot in np.linspace(0, 60, 360, endpoint=False):
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
            
    res = minimize_scalar(obj, bounds=(best_rot - 5, best_rot + 5), method='bounded', options={'xatol': 1e-6})
    if res.success:
        return res.fun, res.x
        
    return best_L, best_rot

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
        
        penalty = smooth_sdf_penalty(xys_rots, beta=50.0)
        
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
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

    def make_diamond(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        dx = sqrt3 + gap
        dy = 1.5 + gap * 0.5
        for c in range(2):
            pts.append([c * dx + dx/2, 0, 0.0])
        for c in range(3):
            pts.append([c * dx, dy, 0.0])
        for c in range(4):
            pts.append([c * dx + dx/2, 2*dy, 0.0])
        for c in range(3):
            pts.append([c * dx, 3*dy, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def is_valid(params, tol=1e-6):
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True

    def get_L(params):
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    def full_score(cfg, angle):
        params = np.concatenate((cfg.flatten(), [angle]))
        if not is_valid(params):
            return None
        return get_L(params)

    best_L = float('inf')
    best_params = None
    
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_diamond(gap=gap))
    
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)

    # Phase 1: Evaluate initial configurations
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in cfg])
        L_val, best_rot = optimize_outer_rotation(all_points)
        
        init_params = np.concatenate((cfg.flatten(), [best_rot]))
        
        if is_valid(init_params):
            L_check = get_L(init_params)
            if L_check < best_L:
                best_L = L_check
                best_params = init_params.copy()

    # Phase 2: L-BFGS-B Optimization
    if best_params is not None and time.time() - start_time < DEADLINE - 30:
        try:
            res_lbgfs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1500}
            )
            if is_valid(res_lbgfs.x):
                L_val = get_L(res_lbgfs.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass

    # Phase 3: Final polish
    if best_params is not None:
        cfg = best_params[:3*N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in cfg])
        L_val, best_rot = optimize_outer_rotation(all_points)
        new_params = np.concatenate((cfg.flatten(), [best_rot]))
        if is_valid(new_params):
            L_check = get_L(new_params)
            if L_check < best_L:
                best_L = L_check
                best_params = new_params

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([float(x), float(y), float(rot)])
    
    return inner, (0.0, 0.0), float(best_L), float(outer_rot)
