import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    angles = np.deg2rad(angle_deg) + np.deg2rad([0, 60, 120, 180, 240, 300])
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _axes_from_polygon(verts):
    """Return edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        axes.append(normal)
    return np.array(axes)

def _separating_axis_check(verts1, verts2, eps=1e-7):
    """Check if two convex polygons are separated (disjoint). Returns True if disjoint."""
    axes1 = _axes_from_polygon(verts1)
    axes2 = _axes_from_polygon(verts2)
    axes = np.vstack([axes1, axes2])
    
    for axis in axes:
        min1, max1 = _project_polygon(verts1, axis)
        min2, max2 = _project_polygon(verts2, axis)
        # Check for separation
        if max1 < min2 - eps or max2 < min1 - eps:
            return True
    return False

def _min_overlap_depth(verts1, verts2):
    """Return minimum translation distance to separate two convex polygons. 0 if disjoint."""
    axes1 = _axes_from_polygon(verts1)
    axes2 = _axes_from_polygon(verts2)
    axes = np.vstack([axes1, axes2])
    
    min_depth = float('inf')
    
    for axis in axes:
        min1, max1 = _project_polygon(verts1, axis)
        min2, max2 = _project_polygon(verts2, axis)
        
        if max1 < min2 or max2 < min1:
            return 0.0  # Disjoint
        
        # Overlap on this axis
        depth = min(max1, max2) - max(min1, min2)
        min_depth = min(min_depth, depth)
    
    return max(0, min_depth)

def _is_valid(inner, eps=1e-7):
    """Check if all pairs of inner hexagons are disjoint."""
    n = len(inner)
    verts_list = [_hex_vertices(cx, cy, a) for cx, cy, a in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if not _separating_axis_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def _outer_hex_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length for given center and orientation."""
    outer_angle = np.deg2rad(outer_angle_deg)
    norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    
    # For each normal direction, find max support of all inner hexagons
    h = np.zeros(6)
    for cx, cy, a_deg in inner:
        verts = _hex_vertices(cx, cy, a_deg)
        for k in range(6):
            proj = np.dot(verts, normals[k])
            h[k] = max(h[k], proj.max())
    
    # Outer hexagon support in direction k: side * sqrt(3)/2 + center . n_k
    # Need: side * sqrt(3)/2 + center . n_k >= h[k]
    # For opposite direction k+3: side * sqrt(3)/2 - center . n_k >= h[k+3]
    # Sum: side * sqrt(3) >= h[k] + h[k+3]
    # So side >= max over k=0,1,2 of (h[k] + h[k+3]) / sqrt(3)
    
    # But we can optimize center position
    # Let's try center at centroid first
    cx_avg = np.mean([p[0] for p in inner])
    cy_avg = np.mean([p[1] for p in inner])
    center = np.array([cx_avg, cy_avg])
    
    # Adjust h by subtracting center contribution
    center_projs = center @ normals.T
    h_adj = h - center_projs
    
    side = max(h_adj[0] + h_adj[3], h_adj[1] + h_adj[4], h_adj[2] + h_adj[5]) / np.sqrt(3)
    
    return max(0.0, side)

def _compute_min_side_full(inner):
    """Find minimum outer hexagon side by trying multiple orientations."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    cx_avg = np.mean([p[0] for p in inner])
    cy_avg = np.mean([p[1] for p in inner])
    
    # Try multiple outer orientations
    for angle_deg in np.linspace(0, 60, 13):
        outer_angle = np.deg2rad(angle_deg)
        norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
        
        # Compute max support for each direction
        h = np.zeros(6)
        for cx, cy, a_deg in inner:
            verts = _hex_vertices(cx, cy, a_deg)
            for k in range(6):
                proj = np.dot(verts, normals[k])
                h[k] = max(h[k], proj.max())
        
        # Optimize center: minimize max over k of (h[k] - c.n_k + h[k+3] + c.n_k) / sqrt(3)
        # = (h[k] + h[k+3]) / sqrt(3), which is independent of center!
        # So center doesn't affect the minimum side for fixed orientation.
        
        side_candidate = max(h[0] + h[3], h[1] + h[4], h[2] + h[5]) / np.sqrt(3)
        
        if side_candidate < best_side - 1e-10:
            best_side = side_candidate
            best_angle = angle_deg
    
    # Refine center for best angle
    if best_side < float('inf'):
        outer_angle = np.deg2rad(best_angle)
        norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
        
        h = np.zeros(6)
        for cx, cy, a_deg in inner:
            verts = _hex_vertices(cx, cy, a_deg)
            for k in range(6):
                proj = np.dot(verts, normals[k])
                h[k] = max(h[k], proj.max())
        
        # Find center that balances the constraints
        # We want c such that for k=0,1,2:
        # c.n_k >= h[k] - side*sqrt(3)/2
        # c.n_k <= side*sqrt(3)/2 - h[k+3]
        # Use midpoint: c.n_k = (h[k] - h[k+3]) / 2
        
        # Solve for c from 3 linear equations
        # n_0 . c = (h[0] - h[3])/2
        # n_1 . c = (h[1] - h[4])/2
        # n_2 . c = (h[2] - h[5])/2
        
        A = np.vstack([normals[0], normals[1], normals[2]])
        b = np.array([(h[0] - h[3])/2, (h[1] - h[4])/2, (h[2] - h[5])/2])
        
        try:
            center = np.linalg.solve(A, b)
            best_center = (center[0], center[1])
        except:
            best_center = (cx_avg, cy_avg)
    
    return best_side, best_center, best_angle

def _smooth_collision_penalty(inner, beta=50.0):
    """Smooth penalty for overlapping inner hexagons."""
    n = len(inner)
    verts_list = [_hex_vertices(cx, cy, a) for cx, cy, a in inner]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            depth = _min_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                penalty += depth * depth
    return penalty

def _smooth_outer_side(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth version of outer hexagon side using log-sum-exp."""
    outer_angle = np.deg2rad(outer_angle_deg)
    norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    
    h = np.zeros(6)
    for cx, cy, a_deg in inner:
        verts = _hex_vertices(cx, cy, a_deg)
        for k in range(6):
            proj = np.dot(verts, normals[k])
            h[k] = max(h[k], proj.max())
    
    # side = max(h[0]+h[3], h[1]+h[4], h[2]+h[5]) / sqrt(3)
    # Use log-sum-exp for smooth max
    vals = np.array([h[0] + h[3], h[1] + h[4], h[2] + h[5]])
    if beta > 0:
        log_vals = vals / beta
        max_log = log_vals.max()
        smooth_max = beta * (max_log + np.log(np.sum(np.exp(log_vals - max_log))))
    else:
        smooth_max = vals.max()
    
    return smooth_max / np.sqrt(3)

def construct():
    t0 = time.time()
    
    # Initial configuration: honeycomb pattern
    dy = np.sqrt(3)
    initial_centers = []
    initial_angles = []
    
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * dy
            initial_centers.append((x - 1.5, y - dy))
            initial_angles.append(0.0)
    
    params0 = np.array([c[0] for c in initial_centers] + [c[1] for c in initial_centers])
    
    inner0 = [(initial_centers[i][0], initial_centers[i][1], initial_angles[i]) for i in range(N)]
    side0, center0, angle0 = _compute_min_side_full(inner0)
    best_side = side0
    best_params = params0.copy()
    best_center = center0
    best_angle = angle0
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        return [(inner0[i][0], inner0[i][1], inner0[i][2]) for i in range(N)], best_center, best_side, best_angle
    
    def objective(params):
        centers = params.reshape(N, 2)
        inner = [(centers[i, 0], centers[i, 1], initial_angles[i]) for i in range(N)]
        
        # Check validity
        if not _is_valid(inner, eps=1e-6):
            # Add penalty for overlaps
            penalty = _smooth_collision_penalty(inner, beta=20.0)
            side = _outer_hex_side(inner, center0, angle0)
            return side + penalty * 10.0
        
        side, _, _ = _compute_min_side_full(inner)
        return side
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, params0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            centers = res.x.reshape(N, 2)
            inner = [(centers[i, 0], centers[i, 1], initial_angles[i]) for i in range(N)]
            _, best_center, best_angle = _compute_min_side_full(inner)
    except Exception:
        pass
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 10:
        centers = best_params.reshape(N, 2)
        inner = [(centers[i, 0], centers[i, 1], initial_angles[i]) for i in range(N)]
        return [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)], best_center, best_side, best_angle
    
    # Polish pass with tighter tolerance
    try:
        res2 = minimize(objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res2.fun < best_side:
            best_side = res2.fun
            best_params = res2.x
            centers = res2.x.reshape(N, 2)
            inner = [(centers[i, 0], centers[i, 1], initial_angles[i]) for i in range(N)]
            _, best_center, best_angle = _compute_min_side_full(inner)
    except Exception:
        pass
    
    centers = best_params.reshape(N, 2)
    inner = [(centers[i, 0], centers[i, 1], initial_angles[i]) for i in range(N)]
    return [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)], best_center, best_side, best_angle
