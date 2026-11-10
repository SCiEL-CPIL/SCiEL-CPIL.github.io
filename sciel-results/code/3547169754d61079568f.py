# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _max_overlap(v1, v2, a1_deg, a2_deg):
    """Return maximum overlap depth across all 12 normals (positive = intersecting)."""
    a1 = np.deg2rad(a1_deg)
    a2 = np.deg2rad(a2_deg)
    normals1 = a1 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals2 = a2 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    
    all_normals = np.vstack([normals1, normals2])
    cos_n = np.cos(all_normals)
    sin_n = np.sin(all_normals)
    
    max_overlap = 0.0
    for k in range(12):
        n = np.array([cos_n[k], sin_n[k]])
        p1 = v1 @ n
        p2 = v2 @ n
        overlap = min(np.max(p2) - np.min(p1), np.max(p1) - np.min(p2))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _min_enclosing_hex_side(vertices_all, outer_angle_deg=0.0):
    """Minimum side of regular hex (centered at origin) enclosing all vertices."""
    oa = np.deg2rad(outer_angle_deg)
    normals = oa + np.deg2rad(30.0 + 60.0 * np.arange(6))
    cos_n = np.cos(normals)
    sin_n = np.sin(normals)
    
    min_side = 0.0
    for p in vertices_all:
        projections = p[0] * cos_n + p[1] * sin_n
        abs_projections = np.abs(projections)
        required_s = np.max(abs_projections) / (np.sqrt(3.0) / 2.0)
        if required_s > min_side:
            min_side = required_s
    return min_side

def _compute_objective(params):
    """
    params: 10 values representing 5 pairs of (x, y).
    The 11 centers are:
    (0,0)
    (x1, y1), (-x1, -y1)
    (x2, y2), (-x2, -y2)
    (x3, y3), (-x3, -y3)
    (x4, y4), (-x4, -y4)
    (x5, y5), (-x5, -y5)
    All angles are 0.
    """
    centers = np.zeros((11, 2))
    centers[0] = [0.0, 0.0]
    
    for i in range(5):
        x = params[2*i]
        y = params[2*i + 1]
        centers[2*i + 1] = [x, y]
        centers[2*i + 2] = [-x, -y]
        
    angles = np.zeros(11)
    
    vertices_list = []
    for i in range(11):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        vertices_list.append(v)
        
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            overlap = _max_overlap(vertices_list[i], vertices_list[j], angles[i], angles[j])
            if overlap > 0:
                penalty += overlap ** 2
                
    all_vertices = np.vstack(vertices_list)
    outer_side = _min_enclosing_hex_side(all_vertices, 0.0)
    
    return outer_side + 100.0 * penalty

def hexagon_packing_11():
    t_start = time.time()
    deadline = t_start + 570.0
    
    # Initial guess based on hexagonal grid
    sqrt3 = np.sqrt(3.0)
    x0 = np.array([
        2.0, 0.0,
        1.0, sqrt3,
        1.0, -sqrt3,
        3.0, sqrt3,
        3.0, -sqrt3
    ])
    
    best_x = x0.copy()
    best_val = _compute_objective(x0)
    
    # Use Nelder-Mead for small dimensional problems
    # Also try a few random restarts if time permits
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            x0_trial = x0.copy()
        else:
            rng = np.random.default_rng(100 + r)
            # Perturb initial guess
            x0_trial = x0 + rng.normal(0, 0.5, size=x0.shape)
            
        try:
            result = minimize(
                _compute_objective, x0_trial,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_x = result.x
        except Exception:
            pass
            
    # Reconstruct full solution
    centers = np.zeros((11, 2))
    centers[0] = [0.0, 0.0]
    for i in range(5):
        x = best_x[2*i]
        y = best_x[2*i + 1]
        centers[2*i + 1] = [x, y]
        centers[2*i + 2] = [-x, -y]
        
    angles = np.zeros(11)
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    vertices_list = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(11)]
    all_vertices = np.vstack(vertices_list)
    outer_side = _min_enclosing_hex_side(all_vertices, 0.0)
    
    return inner_hex_data, outer_hex_data, float(outer_side)

# EVOLVE-BLOCK-END
