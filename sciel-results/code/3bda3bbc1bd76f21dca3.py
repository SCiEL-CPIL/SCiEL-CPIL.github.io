import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Returns 6 vertices of a regular hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append([x, y])
    return np.array(vertices)

def _polygon_overlap_depth(p1, p2):
    """
    Calculates the penetration depth of two convex polygons using Separating Axis Theorem.
    Returns 0 if disjoint, otherwise the minimum distance to separate them (positive value).
    """
    def get_normals(p):
        normals = []
        for i in range(len(p)):
            p_curr = p[i]
            p_next = p[(i + 1) % len(p)]
            edge = p_next - p_curr
            norm = np.array([edge[1], -edge[0]])
            norm_norm = np.linalg.norm(norm)
            if norm_norm > 1e-9:
                normals.append(norm / norm_norm)
        return normals

    normals1 = get_normals(p1)
    normals2 = get_normals(p2)
    
    min_overlap = float('inf')
    
    for normal in normals1 + normals2:
        proj1 = p1 @ normal
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        proj2 = p2 @ normal
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _required_outer_side(points, phi_deg):
    """
    Calculates the minimum side length S of a regular hexagon (centered at origin, rotation phi)
    that contains all points.
    """
    phi_rad = np.deg2rad(phi_deg)
    max_proj = 0.0
    for k in range(6):
        angle = phi_rad + k * np.pi / 3
        u = np.array([np.cos(angle), np.sin(angle)])
        projs = points @ u
        max_abs = np.max(np.abs(projs))
        if max_abs > max_proj:
            max_proj = max_abs
            
    return max_proj * 2.0 / np.sqrt(3)

def _evaluate_config(r1, r2, delta, theta1, theta2, phi_outer):
    """
    Evaluates a specific configuration.
    Returns (S_req, max_overlap, inner_hexes_list)
    """
    centers = []
    angles = []
    
    for k in range(6):
        angle_pos = k * np.pi / 3
        cx = r1 * np.cos(angle_pos)
        cy = r1 * np.sin(angle_pos)
        centers.append([cx, cy])
        angles.append(np.deg2rad(theta1))
        
    for k in range(6):
        angle_pos = delta + k * np.pi / 3
        cx = r2 * np.cos(angle_pos)
        cy = r2 * np.sin(angle_pos)
        centers.append([cx, cy])
        angles.append(np.deg2rad(theta2))
        
    centers = np.array(centers)
    angles = np.array(angles)
    
    all_vertices = []
    hex_polygons = []
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, np.rad2deg(angles[i]))
        hex_polygons.append(verts)
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    S_req = _required_outer_side(all_vertices, phi_outer)
    
    max_overlap = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            depth = _polygon_overlap_depth(hex_polygons[i], hex_polygons[j])
            if depth > max_overlap:
                max_overlap = depth
                
    return S_req, max_overlap, centers, angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def objective(params):
        r1, r2, delta, theta1, theta2, phi_outer = params
        S_req, max_overlap, _, _ = _evaluate_config(r1, r2, delta, theta1, theta2, phi_outer)
        # Penalty for overlap
        penalty = max_overlap ** 2
        return S_req + 1000.0 * penalty

    best_valid_S = float('inf')
    best_valid_params = None
    best_invalid_score = float('inf')
    best_invalid_params = None

    inits = [
        [np.sqrt(3), 2*np.sqrt(3), 0.0, 0.0, 0.0, 0.0],
        [np.sqrt(3), 2*np.sqrt(3), np.pi/6, 0.0, 0.0, 0.0],
        [1.7, 3.5, 0.0, np.pi/6, 0.0, np.pi/6],
        [1.6, 3.4, 0.0, 0.0, np.pi/6, 0.0],
        [1.8, 3.6, np.pi/12, np.pi/12, np.pi/12, np.pi/12]
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness in low dimensions
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            S_req, max_overlap, centers, angles = _evaluate_config(*res.x)
            
            # Check validity
            if max_overlap < 1e-5:
                if S_req < best_valid_S:
                    best_valid_S = S_req
                    best_valid_params = res.x
            else:
                # Track best invalid in case no valid is found (to potentially improve seed if seed is worse? No, seed is valid)
                score = res.fun
                if score < best_invalid_score:
                    best_invalid_score = score
                    best_invalid_params = res.x
                    
        except Exception as e:
            pass
            
    if best_valid_params is not None:
        r1, r2, delta, theta1, theta2, phi_outer = best_valid_params
        _, _, centers, angles = _evaluate_config(r1, r2, delta, theta1, theta2, phi_outer)
        
        inner = []
        for i in range(12):
            inner.append([centers[i, 0], centers[i, 1], np.rad2deg(angles[i])])
            
        return inner, (0.0, 0.0), best_valid_S, phi_outer

    # Fallback to seed
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
