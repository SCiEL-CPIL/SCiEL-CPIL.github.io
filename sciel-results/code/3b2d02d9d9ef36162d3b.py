"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 12

# Helper: vertices of a regular hexagon with side length s, center (cx, cy), rotation angle (radians)
def hex_vertices(cx, cy, s, angle):
    # Regular hexagon vertices at angles 0, 60, ..., 300 degrees (flat-top or pointy-top depending on convention)
    # Using pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees relative to x-axis? 
    # Let's use standard: vertices at angles k*60 degrees for k=0..5, offset by 30 for flat-top?
    # A regular hexagon with side length s has circumradius = s.
    # Vertices at angles: angle + k * pi/3 for k in 0..5
    angles = angle + np.arange(6) * (np.pi / 3.0)
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

# Helper: compute the minimum outer side length needed to contain a set of inner hexagons
# given the outer center and orientation.
def compute_min_outer_side(inner_hexes, outer_center, outer_angle):
    """
    inner_hexes: list of (x, y, angle_degrees)
    outer_center: (cx, cy)
    outer_angle: rotation angle in degrees for the outer hexagon
    
    Returns: minimum side length s such that all inner hexagons are inside the outer hexagon.
    """
    cx, cy = outer_center
    outer_rad = np.radians(outer_angle)
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, np.radians(iang))
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
    
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon is defined by 6 half-planes.
    # A regular hexagon with center (cx, cy), side length s, rotation outer_rad.
    # The distance from center to each side (apothem) is s * sqrt(3) / 2.
    # The normal vectors to the 6 sides point outward at angles: outer_rad + k*pi/3 for k=0..5? 
    # Actually, for a regular hexagon, the normals to the sides are at angles: outer_rad + pi/6 + k*pi/3?
    # Let's think: if vertices are at angles outer_rad + k*pi/3, then the sides are between consecutive vertices.
    # The normal to the side between vertex k and k+1 is at angle outer_rad + (k+0.5)*pi/3.
    # So normals are at: outer_rad + pi/6, outer_rad + pi/6 + pi/3, ..., i.e., outer_rad + pi/6 + k*pi/3 for k=0..5.
    
    normal_angles = outer_rad + np.pi/6 + np.arange(6) * (np.pi / 3.0)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For a point p to be inside the hexagon, we need:
    # dot(p - center, normal) <= apothem = s * sqrt(3) / 2
    # So s >= 2 / sqrt(3) * dot(p - center, normal) for all p and all normals where dot > 0.
    
    rel = all_verts - np.array([cx, cy])
    dots = rel @ normals.T  # shape: (num_verts, 6)
    
    # For each normal direction, find the max dot product
    max_dots = np.max(dots, axis=0)  # shape: (6,)
    
    # s must satisfy: s * sqrt(3)/2 >= max_dots[i] for each i
    # So s >= max_dots[i] * 2 / sqrt(3)
    required_s = max_dots * 2.0 / np.sqrt(3.0)
    
    return np.max(required_s)

# Objective function for optimization
def objective(params):
    """
    params: flattened array of [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, outer_angle]
    where a is in radians.
    """
    n = N
    inner_params = params[:n*3]
    outer_params = params[n*3:]
    
    inner_hexes = []
    for i in range(n):
        x = inner_params[i*3]
        y = inner_params[i*3+1]
        a = inner_params[i*3+2]
        inner_hexes.append((x, y, np.degrees(a)))
    
    cx = outer_params[0]
    cy = outer_params[1]
    outer_angle = np.degrees(outer_params[2])
    
    # Check for overlaps between inner hexagons
    # For simplicity, we use a penalty based on distance between centers.
    # Two unit hexagons overlap if the distance between centers is less than some threshold.
    # The minimum distance for non-overlapping hexagons depends on orientation.
    # A simple lower bound: if centers are closer than sqrt(3), they might overlap (apothem*2 = sqrt(3)).
    # But this is not tight. Let's use a penalty that increases as centers get closer.
    
    overlap_penalty = 0.0
    for i in range(n):
        xi, yi, _ = inner_hexes[i]
        for j in range(i+1, n):
            xj, yj, _ = inner_hexes[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            # If dist < sqrt(3), they likely overlap. Penalty = (sqrt(3) - dist)^2 if positive
            if dist < np.sqrt(3):
                overlap_penalty += (np.sqrt(3) - dist)**2
    
    # Compute required outer side length
    s = compute_min_outer_side(inner_hexes, (cx, cy), outer_angle)
    
    return s + 10.0 * overlap_penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: place hexagons in a grid pattern
    # For 12 hexagons, maybe 3x4 or 4x3 arrangement
    # Let's start with a linear arrangement and let optimization fix it
    
    np.random.seed(42)
    
    # Better initial guess: honeycomb-like arrangement
    # In a honeycomb, hexagons tile the plane.
    # Distance between adjacent centers in honeycomb: 2 * apothem = sqrt(3)
    # Let's place them in a hexagonal grid pattern
    
    initial_inner = []
    # 12 hexagons: maybe 3 rows of 4, or some other arrangement
    # Let's try a compact arrangement
    
    # Hex grid coordinates to Cartesian:
    # For pointy-top hexagons with side length 1:
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row
    
    positions = []
    # Try a 3x4 grid in hex coordinates
    for row in range(3):
        for col in range(4):
            x = np.sqrt(3) * (col + row/2.0)
            y = 1.5 * row
            positions.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = np.mean(xs)
    cy0 = np.mean(ys)
    
    initial_inner = [[x - cx0, y - cy0, a] for (x, y, a) in positions]
    
    # Flatten parameters
    params0 = []
    for (x, y, a) in initial_inner:
        params0.extend([x, y, np.radians(a)])
    params0.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    # Optimize
    best_result = None
    best_score = np.inf
    
    # Try a few different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            p0 = np.array(params0)
        else:
            # Add some random perturbation
            p0 = np.array(params0) + np.random.normal(0, 0.5, size=len(params0))
        
        try:
            result = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_result = result
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to initial guess
        inner = initial_inner
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract best result
    params = best_result.x
    n = N
    inner = []
    for i in range(n):
        x = params[i*3]
        y = params[i*3+1]
        a = np.degrees(params[i*3+2])
        inner.append([x, y, a])
    
    cx = params[n*3]
    cy = params[n*3+1]
    outer_angle = np.degrees(params[n*3+2])
    
    # Compute the actual side length (without penalty)
    s = compute_min_outer_side(inner, (cx, cy), outer_angle)
    
    return inner, (cx, cy), s, outer_angle
# EVOLVE-BLOCK-END
