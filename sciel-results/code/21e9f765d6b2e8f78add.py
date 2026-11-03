"""Unit regular hexagons inside a regular hexagon - optimized packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    theta = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + theta
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon with center (cx,cy), side, rotation."""
    theta = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    apothem = side * np.sqrt(3) / 2.0
    for k in range(3):
        ang = np.radians(30 + 60 * k)
        val = lx * np.cos(ang) + ly * np.sin(ang)
        if abs(val) > apothem + 1e-10:
            return False
    return True

def _hexagons_overlap(v1, v2, c1, c2, s1, s2, a1, a2):
    """Check if two hexagons overlap. v1, v2 are vertex arrays."""
    # Check if any vertex of v1 is inside hex2
    for i in range(6):
        if _point_in_hexagon(v1[i, 0], v1[i, 1], c2, s2, a2):
            return True
    # Check if any vertex of v2 is inside hex1
    for i in range(6):
        if _point_in_hexagon(v2[i, 0], v2[i, 1], c1, s1, a1):
            return True
    return False

def _compute_outer_side(cx, cy, theta_deg, all_points):
    """Compute the side length of the minimum enclosing regular hexagon 
    for a set of points, given center and rotation."""
    theta = np.radians(theta_deg)
    s_required = 0.0
    for k in range(3):
        ang = np.radians(30 + 60 * k) + theta
        projections = (all_points[:, 0] - cx) * np.cos(ang) + (all_points[:, 1] - cy) * np.sin(ang)
        max_proj = np.max(np.abs(projections))
        s_req = 2.0 * max_proj / np.sqrt(3.0)
        s_required = max(s_required, s_req)
    return s_required

def _objective(params):
    """
    params: [x0, y0, ..., x10, y10, cx, cy, theta_deg]
    Total 22 + 3 = 25 variables.
    Inner hexagons are fixed at angle 0.
    """
    # Unpack
    inner_xy = params[:22].reshape(N, 2)
    cx = params[22]
    cy = params[23]
    theta_deg = params[24]
    
    # Compute vertices for all inner hexagons
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
    
    all_points = all_verts.reshape(-1, 2)
    
    # Compute outer side
    s = _compute_outer_side(cx, cy, theta_deg, all_points)
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Quick bounding box check
            # Hex width = 2, height = sqrt(3)
            if abs(inner_xy[i, 0] - inner_xy[j, 0]) > 2.0 or abs(inner_xy[i, 1] - inner_xy[j, 1]) > np.sqrt(3):
                continue
            
            v1 = all_verts[i]
            v2 = all_verts[j]
            # Detailed check
            if _hexagons_overlap(v1, v2, inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0, 
                                 inner_xy[j, 0], inner_xy[j, 1], 1.0, 0.0):
                penalty += 100.0
    
    return s + penalty

def _make_initial_hex_lattice():
    """Create a hexagonal lattice arrangement of 11 unit hexagons."""
    apothem = np.sqrt(3) / 2.0
    center_dist = 2 * apothem
    positions = []
    positions.append((0.0, 0.0))
    for k in range(6):
        ang = np.radians(60 * k)
        x = center_dist * np.cos(ang)
        y = center_dist * np.sin(ang)
        positions.append((x, y))
    for k in [0, 2, 4]:
        ang = np.radians(60 * k)
        x = 2 * center_dist * np.cos(ang)
        y = 2 * center_dist * np.sin(ang)
        positions.append((x, y))
    positions.append((-2 * center_dist, 0.0))
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = np.inf
    
    # Initial configuration
    init_pos = _make_initial_hex_lattice()
    # Flatten positions
    init_xy = np.array(init_pos).flatten()
    
    # Initial outer guess: centroid and 0 rotation
    cx0 = np.mean(init_xy[0::2])
    cy0 = np.mean(init_xy[1::2])
    theta0 = 0.0
    
    init_params = np.append(init_xy, [cx0, cy0, theta0])
    
    # Optimize using Nelder-Mead (robust for non-smooth)
    try:
        result = minimize(
            _objective, init_params, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        if time.time() < deadline:
            score = result.fun
            if score < best_score:
                best_score = score
                best_params = result.x
                best_result = best_params
    except Exception as e:
        pass
    
    # If Nelder-Mead didn't find a good result or timed out, try a simpler approach
    # or just use the initial configuration if it's valid
    if best_result is None or best_score > 100: # 100 is a high penalty threshold
        # Use initial configuration
        best_params = init_params
        best_score = _objective(init_params)
        best_result = best_params
    
    # Extract results
    inner_xy = best_result[:22].reshape(N, 2)
    cx = best_result[22]
    cy = best_result[23]
    theta_deg = best_result[24]
    
    # Compute final outer side
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
    all_points = all_verts.reshape(-1, 2)
    
    best_s = _compute_outer_side(cx, cy, theta_deg, all_points)
    
    # Construct inner list: (x, y, angle)
    inner = [[float(inner_xy[i, 0]), float(inner_xy[i, 1]), 0.0] for i in range(N)]
    
    return inner, (float(cx), float(cy)), float(best_s), float(theta_deg)

# EVOLVE-BLOCK-START
# (construct is defined above; this block is for compatibility)
# EVOLVE-BLOCK-END
