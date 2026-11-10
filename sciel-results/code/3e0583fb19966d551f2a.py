import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3)

def hex_vertices(cx, cy, angle):
    """
    Returns the 6 vertices of a unit hexagon centered at (cx, cy)
    with rotation 'angle' (radians).
    """
    # Vertices of a unit hexagon (side 1) are at distance 1 from center.
    # Angles: 0, 60, 120, 180, 240, 300
    # If we rotate by 'angle', we add 'angle' to these.
    # Standard orientation: flat-top? 
    # If angle=0, vertex at 0? 
    # Let's assume vertex at angle 0 for angle=0.
    # This corresponds to "pointy-top" if y is up? 
    # Doesn't matter as long as consistent.
    base_angles = np.arange(6) * np.pi / 3
    angles = base_angles + angle
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def min_L_required(vertices):
    """
    Calculates the minimum side length L of a regular hexagon centered at origin
    that contains all given vertices.
    """
    if len(vertices) == 0:
        return 0.0
    
    # Polar coordinates
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])
    
    # Normalize theta to [0, 2*pi)
    theta = theta % (2 * np.pi)
    
    # Map to [0, pi/3)
    theta_mod = theta % (np.pi / 3)
    
    # Angle to closest normal
    # Normals are at 0, pi/3, 2pi/3...
    # In [0, pi/3), the closest normal is either 0 or pi/3.
    # The bisector is pi/6.
    # Angle to normal 0 is theta_mod.
    # Angle to normal pi/3 is pi/3 - theta_mod.
    # We want the minimum angle.
    angle_to_normal = np.minimum(theta_mod, np.pi/3 - theta_mod)
    
    # L_req = (2 * r / sqrt(3)) * cos(angle_to_normal)
    # Note: cos is positive in [0, pi/6]
    cos_term = np.cos(angle_to_normal)
    
    L_req = (2 * r / SQRT3) * cos_term
    
    return np.max(L_req)

def overlap_penalty(positions):
    """
    Calculates penalty for overlapping inner hexagons.
    Assumes aligned hexagons. Min distance is sqrt(3).
    """
    n = len(positions)
    penalty = 0.0
    min_dist_sq = 3.0 # (sqrt(3))^2
    
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                # Quadratic penalty
                diff = min_dist_sq - dist_sq
                penalty += diff * diff
    return penalty

def objective(params):
    """
    params: [x1, y1, ..., x12, y12, alpha]
    """
    positions = np.array(params[:24]).reshape(12, 2)
    alpha = params[24]
    
    # Rotate positions by alpha? 
    # No, alpha is the rotation of the hexagons themselves.
    # But if we rotate the hexagons, the vertices change.
    # The centers stay the same.
    # The overlap constraint depends on the relative angle.
    # If all have same angle alpha, relative angle is 0.
    # So min distance is still sqrt(3).
    
    # Calculate L required
    all_vertices = []
    for i in range(12):
        cx, cy = positions[i]
        verts = hex_vertices(cx, cy, alpha)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    L = min_L_required(all_vertices)
    
    # Overlap penalty
    pen = overlap_penalty(positions)
    
    return L + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. 3x4 Grid
    # Lattice vectors
    u = np.array([SQRT3, 0])
    v = np.array([SQRT3/2, 1.5])
    
    # Generate 3x4 grid
    grid_points = []
    for i in range(3):
        for j in range(4):
            # Center of grid at 0
            # i: 0,1,2 -> -1, 0, 1
            # j: 0,1,2,3 -> -1.5, -0.5, 0.5, 1.5
            p = (i-1)*u + (j-1.5)*v
            grid_points.append(p)
    
    initial_params = np.array(grid_points).flatten()
    initial_params = np.append(initial_params, 0.0) # alpha
    
    # Other initial guesses: Randomized
    guesses = [initial_params]
    
    # Add a few random perturbations
    for _ in range(10):
        p = initial_params.copy()
        p[:24] += np.random.normal(0, 0.5, 24)
        p[24] = np.random.uniform(0, np.pi/3)
        guesses.append(p)
        
    # Run optimization for each guess
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead
            from scipy.optimize import minimize
            
            # Limit iterations to ensure speed
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_score:
                # Check if valid (low penalty)
                pos = res.x[:24].reshape(12, 2)
                pen = overlap_penalty(pos)
                if pen < 1e-3: # Strictly non-overlapping
                    L = objective(res.x) - 100.0 * pen
                    if L < best_score:
                        best_score = L
                        best_params = res.x
        except Exception as e:
            continue
            
    # If no good result, fallback to initial guess
    if best_params is None:
        best_params = initial_params
        best_score = objective(best_params)

    # Extract results
    positions = best_params[:24].reshape(12, 2)
    alpha = best_params[24]
    
    # Format inner hexagons
    inner = []
    for i in range(12):
        x, y = positions[i]
        # Convert angle to degrees
        angle_deg = np.degrees(alpha) % 360
        inner.append([x, y, angle_deg])
        
    # Outer hexagon
    center = [0.0, 0.0]
    side = best_score
    angle_outer = 0.0
    
    return inner, center, side, angle_outer

