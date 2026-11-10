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

def generate_honeycomb_config(scale=1.0):
    """Generate a honeycomb-like configuration of 11 hexagons."""
    h = math.sqrt(3) / 2
    coords = []
    # 4-3-4 honeycomb pattern
    for i in range(4):
        coords.append([i*3.0*scale, 0.0])
    for i in range(3):
        coords.append([1.5*scale + i*3.0*scale, 2*h*scale])
    for i in range(4):
        coords.append([i*3.0*scale, 4*h*scale])
    
    # Center it
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    return np.array(coords)

def generate_ring_config():
    """Generate a ring configuration: 1 center + 6 around + 4 outer."""
    h = math.sqrt(3) / 2
    coords = [[0.0, 0.0]]
    # 6 hexagons around the center
    for i in range(6):
        angle = i * 60
        x = 3.0 * math.cos(math.radians(angle))
        y = 3.0 * math.sin(math.radians(angle))
        coords.append([x, y])
    # 4 more hexagons at the outer ring
    for i in range(4):
        angle = 30 + i * 90
        x = 5.0 * math.cos(math.radians(angle))
        y = 5.0 * math.sin(math.radians(angle))
        coords.append([x, y])
    
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    return np.array(coords)

def generate_star_config():
    """Generate a star configuration."""
    h = math.sqrt(3) / 2
    coords = [[0.0, 0.0]]
    # 6 hexagons in a ring
    for i in range(6):
        angle = i * 60 + 30
        x = 2.8 * math.cos(math.radians(angle))
        y = 2.8 * math.sin(math.radians(angle))
        coords.append([x, y])
    # 4 more hexagons
    for i in range(4):
        angle = i * 90
        x = 4.5 * math.cos(math.radians(angle))
        y = 4.5 * math.sin(math.radians(angle))
        coords.append([x, y])
    
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

def optimize_from_config(config, outer_angle, start_time, deadline, safety_margin):
    """Optimize a single configuration and return the best result."""
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = config[:, 0]
    inner_hex_data[:, 1] = config[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    initial_params = inner_hex_data.flatten()
    
    best_side = float('inf')
    best_inner = None
    
    # Try multiple optimization methods and restarts
    for method in ['Nelder-Mead', 'L-BFGS-B']:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        for trial in range(3):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            if trial == 0:
                x_init = initial_params.copy()
            else:
                np.random.seed(trial * 1234 + 5678 + hash(method) % 1000)
                x_init = initial_params + np.random.randn(33) * 0.1
            
            try:
                result = minimize(
                    objective,
                    x_init,
                    args=(outer_angle,),
                    method=method,
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                
                optimized_params = result.x
                optimized_config = optimized_params.reshape(11, 3)
                
                # Check validity
                outer_hex_data = np.array([0.0, 0.0, outer_angle])
                side = calculate_min_side(optimized_config, outer_angle)
                
                if is_valid_configuration(optimized_config, outer_hex_data, side):
                    if side < best_side:
                        best_side = side
                        best_inner = optimized_config.copy()
                        
            except Exception:
                pass
    
    return best_side, best_inner

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0
    
    # Generate multiple initial configurations
    configs = [
        generate_honeycomb_config(1.0),
        generate_honeycomb_config(0.95),
        generate_honeycomb_config(1.05),
        generate_ring_config(),
        generate_star_config(),
    ]
    
    best_side = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    
    # For each configuration, find the best outer angle and optimize
    for config in configs:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Find best outer angle for this configuration
        best_angle_for_config = 0.0
        best_side_for_config = float('inf')
        
        for outer_angle in np.arange(0, 30.01, 1.0):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(np.column_stack([config, np.zeros(11)]), outer_angle)
            if side < best_side_for_config:
                best_side_for_config = side
                best_angle_for_config = outer_angle
        
        # Refine outer angle
        for outer_angle in np.arange(max(0, best_angle_for_config - 1.0), min(30, best_angle_for_config + 1.0) + 0.01, 0.1):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(np.column_stack([config, np.zeros(11)]), outer_angle)
            if side < best_side_for_config:
                best_side_for_config = side
                best_angle_for_config = outer_angle
        
        # Optimize positions
        side_optimized, inner_optimized = optimize_from_config(
            config, best_angle_for_config, start_time, deadline, safety_margin
        )
        
        if inner_optimized is not None:
            if side_optimized < best_side:
                best_side = side_optimized
                best_inner = inner_optimized
                best_outer_angle = best_angle_for_config
    
    # If no valid config found, use the best unoptimized config
    if best_inner is None:
        for config in configs:
            for outer_angle in np.arange(0, 30.01, 1.0):
                side = calculate_min_side(np.column_stack([config, np.zeros(11)]), outer_angle)
                outer_hex_data = np.array([0.0, 0.0, outer_angle])
                if is_valid_configuration(np.column_stack([config, np.zeros(11)]), outer_hex_data, side):
                    if side < best_side:
                        best_side = side
                        best_inner = np.column_stack([config, np.zeros(11)])
                        best_outer_angle = outer_angle
                    break
            if best_inner is not None:
                break
    
    # Final fallback
    if best_inner is None:
        best_inner = np.column_stack([generate_honeycomb_config(1.0), np.zeros(11)])
        best_outer_angle = 0.0
        best_side = calculate_min_side(best_inner, 0.0)
    
    best_outer = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
