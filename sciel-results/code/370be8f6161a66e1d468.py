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
        # Perpendicular to edge
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        # Ensure outward pointing
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(verts1, verts2):
    """
    Compute the minimum gap between two hexagons based on the 12 edge normals.
    For each of the 12 normals (6 from hex1, 6 from hex2), project both hexes.
    Gap = min(max_proj_other - max_proj_self) for separating normals?
    Actually, the condition is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6".
    This means we need to find if there exists a normal n such that:
    max_{p in hex1} dot(p, n) < min_{q in hex2} dot(q, n) - 1e-6
    OR
    max_{p in hex2} dot(p, n) < min_{q in hex1} dot(q, n) - 1e-6
    
    If no such normal exists, they are intersecting (or touching).
    We want to maximize the minimum gap over all pairs, or rather, ensure the gap is > 0.
    For optimization, we can define the 'penalty' as the maximum over all pairs and all normals of the overlap.
    Overlap for normal n between hex A and B:
    max(0, max_proj_A - min_proj_B + 1e-6, max_proj_B - min_proj_A + 1e-6)
    We want this to be <= 0 for all pairs.
    So the constraint violation is max over all pairs and normals of this overlap.
    """
    n1 = get_edge_normals(verts1)
    n2 = get_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    max_violation = 0.0
    
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    # Overlap 1: Hex1 is to the 'left' of Hex2 (max1 < min2 is good)
    # Violation if max1 >= min2 - 1e-6
    viol1 = np.maximum(0, max1 - min2 + 1e-6)
    # Overlap 2: Hex2 is to the 'left' of Hex1
    viol2 = np.maximum(0, max2 - min1 + 1e-6)
    
    total_viol = np.maximum(viol1, viol2)
    max_violation = np.max(total_viol)
    
    return max_violation


def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculates the minimum side length S of an outer regular hexagon centered at origin,
    with given angle, that contains all inner hexagons.
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    # Normals of the outer hexagon (apothem directions)
    normal_angles = [outer_angle_deg + 30 + 60*k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    max_proj = 0.0
    
    # Precompute base vertices for unit hexagon
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    for i in range(inner_hex_data.shape[0]):
        x, y, _ = inner_hex_data[i]
        verts = base_verts + np.array([x, y])
        
        projs = verts @ normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    # Apothem = S * sqrt(3) / 2
    # S = 2 * Apothem / sqrt(3)
    side = 2.0 * max_proj / math.sqrt(3)
    
    return side


def is_valid_configuration(inner_hex_data, outer_hex_data, outer_side, tol=1e-6):
    """Check if the configuration is valid."""
    # 1. Check outer containment
    outer_cx, outer_cy, outer_angle = outer_hex_data
    # Shift coordinates so outer center is at origin for simplicity in check?
    # The problem says "Coordinates are first shifted by the outer center".
    # So we assume inner_hex_data is relative to outer center?
    # "shifted inner hexagons must keep edge lengths..."
    # Let's assume the returned inner_hex_data is in the same coordinate system as outer_hex_data.
    # The evaluator shifts them.
    # For our check, we can just use the raw coords.
    
    # Outer vertices
    outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Check each inner hexagon vertex is inside outer hexagon
    # For a convex polygon, point is inside if it is to the left of all edges (in CCW order)
    # Or using the cross product check from the prompt:
    # "every inner vertex p satisfies cross(p2 - p1, p - p1) >= -1e-6 for each outer edge from vertex p1 to the next vertex p2"
    
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    for i in range(inner_hex_data.shape[0]):
        x, y, _ = inner_hex_data[i]
        verts = base_verts + np.array([x, y])
        
        for v in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j+1)%6]
                e = p2 - p1
                vp = v - p1
                cross = e[0]*vp[1] - e[1]*vp[0]
                if cross < -tol:
                    return False
                    
    # 2. Check inner separation
    for i in range(inner_hex_data.shape[0]):
        for j in range(i+1, inner_hex_data.shape[0]):
            v1 = hexagon_vertices(inner_hex_data[i,0], inner_hex_data[i,1], 1.0, inner_hex_data[i,2])
            v2 = hexagon_vertices(inner_hex_data[j,0], inner_hex_data[j,1], 1.0, inner_hex_data[j,2])
            viol = min_gap_between_hexes(v1, v2)
            if viol > tol:
                return False
                
    return True


def objective(params, outer_angle_deg):
    """
    Objective function for scipy.optimize.minimize.
    params: array of shape (11, 2) for (x, y) of each inner hexagon.
    Returns: outer_side_length + penalty for violations.
    """
    # Reconstruct inner_hex_data
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0:2] = params
    # Angles are fixed to 0 for now? Or should we optimize angles too?
    # Optimizing angles increases dimensionality and complexity.
    # Let's try fixing angles to 0 first, as it's a strong prior.
    # If we want to optimize angles, params would be (11, 3).
    
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    # Penalty for separation violations
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            v1 = hexagon_vertices(inner_hex_data[i,0], inner_hex_data[i,1], 1.0, 0.0)
            v2 = hexagon_vertices(inner_hex_data[j,0], inner_hex_data[j,1], 1.0, 0.0)
            viol = min_gap_between_hexes(v1, v2)
            if viol > 0:
                penalty += viol * 100 # Heavy penalty
    
    return side + penalty


def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    # 1. Initial guess: 4-3-4 honeycomb grid, slightly scaled to ensure initial validity
    h = math.sqrt(3) / 2
    coords = []
    for i in range(4):
        coords.append([i*3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i*3.0, 2*h])
    for i in range(4):
        coords.append([i*3.0, 4*h])
        
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
        
    # Scale to create gaps
    scale = 1.001
    for c in coords:
        c[0] *= scale
        c[1] *= scale
        
    initial_params = np.array(coords)
    
    # 2. Optimize outer angle
    # We can jointly optimize inner positions and outer angle?
    # It's easier to loop over outer angle and optimize inner positions for each.
    # Or add outer angle to the parameter vector.
    # Let's add outer angle to the parameter vector.
    # params = [x1, y1, ..., x11, y11, outer_angle]
    
    def objective_joint(params):
        inner_params = params[:22].reshape(11, 2)
        outer_angle = params[22]
        
        inner_hex_data = np.zeros((11, 3))
        inner_hex_data[:, 0:2] = inner_params
        inner_hex_data[:, 2] = 0.0 # Fixed angle for inner
        
        side = calculate_min_side(inner_hex_data, outer_angle)
        
        penalty = 0.0
        # Separation penalty
        for i in range(11):
            for j in range(i+1, 11):
                v1 = hexagon_vertices(inner_hex_data[i,0], inner_hex_data[i,1], 1.0, 0.0)
                v2 = hexagon_vertices(inner_hex_data[j,0], inner_hex_data[j,1], 1.0, 0.0)
                viol = min_gap_between_hexes(v1, v2)
                if viol > 0:
                    penalty += viol * 100
                    
        # Containment penalty?
        # If side is calculated correctly, containment is guaranteed by definition of side.
        # But we need to ensure the hexagons are actually inside.
        # calculate_min_side finds the MINIMUM side. So if we return that side,
        # the hexagons are exactly touching the boundary.
        # This is valid.
        
        return side + penalty

    initial_full_params = np.append(initial_params.flatten(), 0.0)
    
    # Run optimization
    best_side = float('inf')
    best_inner = initial_params.copy()
    best_outer_angle = 0.0
    
    # Try a few starting angles for the outer hexagon
    # Symmetry suggests 0, 15, 30 might be good.
    start_angles = [0.0, 15.0, 30.0]
    
    for start_angle in start_angles:
        if time.time() - start_time > deadline - safety_margin:
            break
            
        init = np.append(initial_params.flatten(), start_angle)
        
        try:
            res = minimize(objective_joint, init, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_side:
                best_params = res.x
                best_side = res.fun
                
                # Extract best
                best_inner = best_params[:22].reshape(11, 2)
                best_outer_angle = best_params[22]
                
        except Exception as e:
            pass
            
    # Refine the best solution found
    # Use the best_inner and best_outer_angle to do a final check and potentially a finer search
    
    # Construct final data
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0:2] = best_inner
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    # Calculate the true side length for the best configuration
    final_side = calculate_min_side(inner_hex_data, best_outer_angle)
    
    # Add a tiny epsilon to side to ensure strict containment if the evaluator is strict?
    # The evaluator uses -1e-6 tolerance for containment.
    # So we can be tight.
    
    # However, we must ensure separation.
    # If the optimizer didn't fully resolve the penalty, we might have overlaps.
    # Let's do a final validity check.
    if not is_valid_configuration(inner_hex_data, outer_hex_data, final_side):
        # Fallback to the initial scaled grid which is known to be valid
        # (assuming scale 1.001 is enough)
        inner_hex_data = np.zeros((11, 3))
        inner_hex_data[:, 0:2] = initial_params
        inner_hex_data[:, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        final_side = calculate_min_side(inner_hex_data, 0.0)
        
    return inner_hex_data, outer_hex_data, final_side

# EVOLVE-BLOCK-END
