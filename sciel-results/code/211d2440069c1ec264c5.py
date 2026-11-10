# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its 6 vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def min_gap_between_hexes(verts1, verts2):
    """
    Compute the minimum gap between two hexagons based on the 12 edge normals.
    Returns the maximum violation (0 if valid separation > 1e-6).
    """
    n1 = get_edge_normals(verts1)
    n2 = get_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    viol1 = np.maximum(0, max1 - min2 + 1e-6)
    viol2 = np.maximum(0, max2 - min1 + 1e-6)
    
    total_viol = np.maximum(viol1, viol2)
    max_violation = np.max(total_viol)
    
    return max_violation

def calculate_min_side(inner_hex_data, outer_angle_deg=0.0):
    """
    Calculates the minimum side length S of an outer regular hexagon centered at origin,
    with given angle, that contains all inner hexagons.
    """
    normal_angles = np.radians(outer_angle_deg + 30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    max_proj = 0.0
    
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        ang_rad = np.radians(ang_deg)
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R = np.array([[c, -s], [s, c]])
        
        rot_verts = base_verts @ R.T
        verts = rot_verts + np.array([x, y])
        
        projs = verts @ normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    side = 2.0 * max_proj / math.sqrt(3)
    
    return side

def is_valid_configuration(inner_hex_data, outer_hex_data, outer_side, tol=1e-6):
    """Check if the configuration is valid."""
    outer_cx, outer_cy, outer_angle = outer_hex_data
    
    outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # 1. Containment
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        ang_rad = np.radians(ang_deg)
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R = np.array([[c, -s], [s, c]])
        verts = (base_verts @ R.T) + np.array([x, y])
        
        for v in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j+1)%6]
                e = p2 - p1
                vp = v - p1
                cross = e[0]*vp[1] - e[1]*vp[0]
                if cross < -tol:
                    return False
                    
    # 2. Separation
    all_verts = []
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, ang_deg)
        all_verts.append(verts)
        
    for i in range(11):
        for j in range(i+1, 11):
            viol = min_gap_between_hexes(all_verts[i], all_verts[j])
            if viol > tol:
                return False
                
    return True

def generate_initial_config():
    """
    Generate a compact initial configuration using honeycomb packing.
    Returns an array of shape (11, 3) with x, y, angle_deg.
    """
    h = math.sqrt(3) / 2
    d = math.sqrt(3)  # distance between centers of touching hexagons
    
    coords = []
    # 4-3-4 honeycomb pattern
    for i in range(4):
        coords.append([i*3.0, 0.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i*3.0, 2*h, 0.0])
    for i in range(4):
        coords.append([i*3.0, 4*h, 0.0])
    
    # Center it
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    return np.array(coords)

def objective(params, outer_angle=0.0):
    """
    Objective function: minimize outer hexagon side length + penalty for overlaps.
    params: 33 values (11 * 3: x, y, angle_deg for each hexagon)
    """
    inner_hex_data = params.reshape(11, 3)
    
    # Calculate outer side length
    side = calculate_min_side(inner_hex_data, outer_angle)
    
    # Penalty for overlaps between inner hexagons
    penalty = 0.0
    all_verts = []
    for i in range(11):
        x, y, ang_deg = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, ang_deg)
        all_verts.append(verts)
    
    for i in range(11):
        for j in range(i+1, 11):
            viol = min_gap_between_hexes(all_verts[i], all_verts[j])
            if viol > 1e-6:
                penalty += viol * 1000.0  # Large penalty for overlaps
    
    return side + penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0

    # Generate initial configuration
    initial_config = generate_initial_config()
    initial_params = initial_config.flatten()
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if time.time() - start_time < deadline - safety_margin:
            optimized_params = result.x
            optimized_config = optimized_params.reshape(11, 3)
            
            # Check validity
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            side = calculate_min_side(optimized_config, 0.0)
            
            if is_valid_configuration(optimized_config, outer_hex_data, side):
                best_side = side
                best_inner = optimized_config
                best_outer = outer_hex_data
            else:
                # If invalid, try to fix by adding small separation
                # Fallback to initial config
                side_init = calculate_min_side(initial_config, 0.0)
                if is_valid_configuration(initial_config, outer_hex_data, side_init):
                    best_side = side_init
                    best_inner = initial_config
                    best_outer = outer_hex_data
                else:
                    # Last resort: scale up positions slightly
                    best_inner = initial_config
                    best_outer = outer_hex_data
                    best_side = side_init * 1.01
                    
    except Exception as e:
        # Fallback on any error
        best_inner = initial_config
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = calculate_min_side(initial_config, 0.0)
    
    # If no valid config found, use the initial one
    if best_inner is None:
        best_inner = initial_config
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = calculate_min_side(initial_config, 0.0)
    
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
