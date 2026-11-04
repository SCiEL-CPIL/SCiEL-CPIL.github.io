import numpy as np
import time
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

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    # Normals to the sides are at 30 + 60k degrees relative to the vertex directions.
    # If vertices are at rot_deg, sides normals are at rot_deg + 30 + 60k.
    # Wait, let's verify.
    # Vertices at 0, 60, 120...
    # Side between 0 and 60 has normal at 30.
    # So normals are at rot_deg + 30 + 60k.
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = (sqrt(3)/2) * side
    # max_proj is the max distance from center to a side (apothem)
    # So side = 2 * apothem / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(params, beta=50.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    
    # Fast pre-filter: only compute SAT for pairs that might overlap
    centers = xys_rots[:, :2]
    
    # Precompute vertices for all hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    # Compute pairwise penetrations with smooth max
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:  # Only compute SAT if close enough
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    # Use log-sum-exp for smooth approximation of sum of squares
    # sum(p^2) ≈ beta * logsumexp(2*p/beta) for large beta
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:  # Avoid overflow
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Use smooth penalty for better gradient information
        penalty = smooth_overlap_penalty(params, beta=50.0)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5 # Approximate gap in y
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
    
    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
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
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Phase 1: L-BFGS-B with smooth objective to navigate non-convex landscape
            try:
                res_lbgfs = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 500
                    }
                )
                
                # Check if L-BFGS-B result is valid
                if is_valid(res_lbgfs.x):
                    L_val = get_L(res_lbgfs.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_lbgfs.x.copy()
                        
            except Exception:
                pass
            
            # Phase 2: Nelder-Mead for local polishing
            if best_params is not None:
                try:
                    res_nm = minimize(
                        objective,
                        best_params,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 1000,
                            'xatol': 1e-5,
                            'fatol': 1e-5,
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

    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_L, outer_rot

# EVOLVE-BLOCK-END
