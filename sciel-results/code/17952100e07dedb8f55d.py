import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices_batch(xys_rots):
    """
    Batch compute vertices for all hexagons. xys_rots is (N, 3) array: [x, y, angle_deg].
    Returns (N, 6, 2) array of vertices.
    """
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

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (positive), or separation distance if not.
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
            return -gap  # negative = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return min_overlap  # positive = overlapping

def compute_sat_penalties_vectorized(vertices):
    """
    Vectorized computation of penetration depths for all pairs of hexagons using SAT.
    vertices: (N, 6, 2) array of vertices for N hexagons.
    Returns: (N, N) matrix of penetration depths (0 if no overlap, positive if overlapping).
    """
    N_pts = vertices.shape[0]
    if N_pts == 0:
        return np.zeros((0, 0))
    
    # Precompute edge normals for each hexagon
    edges = np.zeros((N_pts, 6, 2))
    for i in range(N_pts):
        for j in range(6):
            p1 = vertices[i, j]
            p2 = vertices[i, (j+1) % 6]
            e = p2 - p1
            edges[i, j] = np.array([e[1], -e[0]])
            norm = np.linalg.norm(edges[i, j])
            if norm > 1e-12:
                edges[i, j] /= norm
    
    penetrations = np.zeros((N_pts, N_pts))
    
    for i in range(N_pts):
        for j in range(i+1, N_pts):
            axes_i = edges[i]
            axes_j = edges[j]
            all_axes = np.vstack([axes_i, axes_j])
            
            proj_i = vertices[i] @ all_axes.T
            min_i = np.min(proj_i, axis=0)
            max_i = np.max(proj_i, axis=0)
            
            proj_j = vertices[j] @ all_axes.T
            min_j = np.min(proj_j, axis=0)
            max_j = np.max(proj_j, axis=0)
            
            separated = (max_i < min_j) | (max_j < min_i)
            
            if np.any(separated):
                penetrations[i, j] = 0.0
                penetrations[j, i] = 0.0
            else:
                overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                min_overlap = np.min(overlaps)
                if min_overlap > 1e-9:
                    penetrations[i, j] = min_overlap
                    penetrations[j, i] = min_overlap
                else:
                    penetrations[i, j] = 0.0
                    penetrations[j, i] = 0.0
    
    return penetrations

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """Compute outer hex side from a list of vertex arrays."""
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(params, beta=50.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    """
    xys_rots = params[:3*N].reshape(N, 3)
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
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat > 0:
                    penetrations.append(d_sat)
    
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
        
        # Smooth penalty
        penalty = smooth_overlap_penalty(params, beta=50.0)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
    
    def is_valid(params, tol=1e-6):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices = get_hex_vertices_batch(xys_rots)
        pen_matrix = compute_sat_penalties_vectorized(vertices)
        iu = np.triu_indices(N, k=1)
        penetrations = pen_matrix[iu]
        return np.all(penetrations < tol)
    
    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        vertices = get_hex_vertices_batch(xys_rots)
        vertices_list = [vertices[i] for i in range(N)]
        return outer_hex_side_batch(vertices_list, outer_rot)
    
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
    
    def make_ring12_rotated(gap=0.0):
        """12 hexagons in a ring, all rotated 30 degrees."""
        pts = make_ring12(gap)
        pts[:, 2] = 30.0
        return pts
    
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
    configs.append(make_ring12_rotated(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:5]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 75.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 600,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices = get_hex_vertices_batch(xys_rots)
                pen_matrix = compute_sat_penalties_vectorized(vertices)
                iu = np.triu_indices(N, k=1)
                penetrations = pen_matrix[iu]
                
                if np.all(penetrations < 1e-6):
                    vertices_list = [vertices[i] for i in range(N)]
                    L_val = outer_hex_side_batch(vertices_list, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Final refinement with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
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
            
            vertices = get_hex_vertices_batch(xys_rots)
            pen_matrix = compute_sat_penalties_vectorized(vertices)
            iu = np.triu_indices(N, k=1)
            penetrations = pen_matrix[iu]
            
            if np.all(penetrations < 1e-6):
                vertices_list = [vertices[i] for i in range(N)]
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        vertices = get_hex_vertices_batch(np.array(inner))
        vertices_list = [vertices[i] for i in range(N)]
        L_val = outer_hex_side_batch(vertices_list, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
