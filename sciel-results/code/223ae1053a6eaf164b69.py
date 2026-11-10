# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def _compute_s(inner_data, outer_angle):
    """Compute the required side length s for a given inner configuration and outer angle."""
    all_verts = _get_all_inner_vertices(inner_data)
    return _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle)

def _penalty_function(params):
    """
    Objective function for optimization.
    params: [x0, y0, ..., x11, y11, outer_angle]
    Returns: s + penalty_for_overlap
    """
    centers = params[:24].reshape(12, 2)
    outer_angle = params[24]
    
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = centers[:, 0]
    inner_data[:, 1] = centers[:, 1]
    inner_data[:, 2] = 0.0 # Keep angles 0 for now
    
    s = _compute_s(inner_data, outer_angle)
    
    # Check for overlaps
    penalty = 0.0
    verts_list = [_hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, 0.0) for i in range(12)]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(verts_list[i], verts_list[j]):
                # Calculate overlap depth approximately
                # For simplicity, use a fixed penalty if they overlap
                penalty += 10.0
    
    return s + penalty

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Initial arrangement: Compact cluster
    # Using a known good configuration for 12 hexagons
    # 1 center, 6 surrounding, 5 more on the perimeter
    # Or simply a 3x4 rhombus shifted to center
    # Let's use the 3x4 rhombus
    # Rows of 3, 4, 3, 2? No.
    # Let's use the "flower" of 7 + 5 extra
    # A better initial guess is a grid
    # Hex grid coordinates
    # Let's just use the previous seed's best known configuration but optimized
    
    # Start with the configuration from the current program's seed
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
        [0, -4.33, 0]
    ])
    
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Initial parameters for optimizer
    x0 = np.zeros(25)
    x0[:24] = inner[:, :2].flatten()
    x0[24] = 0.0 # Initial outer angle
    
    # Use Nelder-Mead for robustness without gradients
    # It might be slow, so we limit iterations
    best_params = x0.copy()
    best_val = _penalty_function(x0)
    
    # Run multiple restarts with different initial angles and small perturbations
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        if restart == 0:
            current_x0 = x0.copy()
        else:
            current_x0 = x0 + np.random.normal(0, 0.5, 25)
            current_x0[24] = np.random.uniform(0, 60)
            
        try:
            res = minimize(_penalty_function, current_x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Extract best solution
    best_centers = best_params[:24].reshape(12, 2)
    best_outer_angle = best_params[24]
    
    best_inner = np.zeros((12, 3))
    best_inner[:, 0] = best_centers[:, 0]
    best_inner[:, 1] = best_centers[:, 1]
    best_inner[:, 2] = 0.0
    
    best_s = _compute_s(best_inner, best_outer_angle)
    
    # Final validation and cleanup
    best_outer = np.array([0, 0, best_outer_angle])
    
    if not _is_valid(best_inner, best_outer, best_s):
        # Fallback to seed if optimization failed
        inner_seed = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        com_seed = np.mean(inner_seed[:, :2], axis=0)
        inner_seed[:, :2] -= com_seed
        best_inner = inner_seed
        best_outer = np.array([0, 0, 0])
        best_s = _compute_s(best_inner, 0)
        
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
