import numpy as np
import math
import time
from scipy.optimize import minimize

N = 12
# Time management
_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _calculate_outer_side(centers, theta_rad):
    """
    Calculate the minimum side length of the outer hexagon (centered at origin, rotated by theta_rad)
    required to contain all inner unit hexagons (also rotated by theta_rad) at given centers.
    
    centers: list of (x, y)
    """
    # Apothem of unit hexagon
    apothem = math.sqrt(3) / 2.0
    
    # Normals for the outer hexagon (and inner, since they are aligned)
    # The normals point outward from the center of the hexagon.
    # For a hexagon with vertices at angles 0, 60, ..., the normals are at 30, 90, ...
    # Wait, standard hexagon with vertex at (1,0) has normal at 30 deg?
    # Vertices: (1,0), (0.5, sqrt(3)/2), ...
    # Edge 1: (1,0) to (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
    # Normal direction: (sqrt(3)/2, 0.5) -> angle 30 deg.
    # So normals are at 30 + 60*k degrees.
    
    max_proj = 0.0
    for k in range(6):
        theta_n = theta_rad + math.radians(30 + 60 * k)
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        
        for cx, cy in centers:
            # The maximum projection of a unit hexagon centered at (cx, cy) onto normal (nx, ny)
            # is the projection of the center plus the apothem.
            proj = cx * nx + cy * ny + apothem
            if proj > max_proj:
                max_proj = proj
    
    # The required apothem of the outer hexagon is max_proj.
    # Side length S = apothem / (sqrt(3)/2)
    return max_proj / apothem

def _collision_penalty(centers, theta_rad, lambda_pen=100.0):
    """
    Calculate smooth penalty for overlaps.
    Two unit hexagons with same orientation overlap if max_k |n_k . (c_i - c_j)| < sqrt(3).
    """
    penalty = 0.0
    threshold = math.sqrt(3)  # 2 * apothem
    
    # Precompute normals
    normals = []
    for k in range(6):
        theta_n = theta_rad + math.radians(30 + 60 * k)
        normals.append((math.cos(theta_n), math.sin(theta_n)))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            
            max_abs_dot = 0.0
            for nx, ny in normals:
                dot = abs(dx * nx + dy * ny)
                if dot > max_abs_dot:
                    max_abs_dot = dot
            
            # If max_abs_dot < threshold, they overlap.
            # Penetration depth is roughly (threshold - max_abs_dot)
            if max_abs_dot < threshold:
                diff = threshold - max_abs_dot
                penalty += diff * diff
    
    return lambda_pen * penalty

def _objective(params):
    """
    Objective function for optimization.
    params: [x0, y0, x1, y1, ..., x11, y11, theta_rad]
    """
    if _time_left() < 1.0:
        return float('inf')
        
    centers = []
    for i in range(N):
        x = params[2*i]
        y = params[2*i+1]
        centers.append((x, y))
    
    theta_rad = params[2*N]
    
    side = _calculate_outer_side(centers, theta_rad)
    pen = _collision_penalty(centers, theta_rad)
    
    return side + pen

def _generate_initial_config():
    """
    Generate a good initial configuration: a subset of the honeycomb lattice.
    """
    # Honeycomb lattice points for unit hexagons (distance between centers in lattice)
    # Distance between adjacent centers in honeycomb is sqrt(3) ~ 1.732
    # We want 12 hexagons.
    # A central hexagon surrounded by 6, and then 5 more from the next ring?
    # Or just a rectangular grid of 3x4?
    
    # Let's use a 3x4 grid with offsets (honeycomb-like)
    # Spacing x: 1.5 * 2 = 3? No, for flat-top hexagons:
    # Width = 2, Height = sqrt(3).
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Offset for odd rows: 0.75
    
    centers = []
    # 4 columns, 3 rows
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            y = r * math.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            # Center the whole thing
            centers.append((x - 2.25, y - math.sqrt(3)))
            
    # Convert to radians for theta (start with 0)
    theta = 0.0
    
    params = []
    for x, y in centers:
        params.append(x)
        params.append(y)
    params.append(theta)
    
    return np.array(params)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score = float('inf')
    
    # Try a few initial configurations
    configs = [
        _generate_initial_config(),
    ]
    
    # Add a random config
    np.random.seed(42)
    rand_centers = []
    for i in range(N):
        rand_centers.append((np.random.uniform(-4, 4), np.random.uniform(-4, 4)))
    rand_params = []
    for x, y in rand_centers:
        rand_params.append(x)
        rand_params.append(y)
    rand_params.append(0.0)
    configs.append(np.array(rand_params))
    
    # Try different initial global rotations
    for base_config in configs:
        if _time_left() < 10.0:
            break
            
        for theta_init in [0.0, math.pi/6, math.pi/3]:
            if _time_left() < 5.0:
                break
                
            params = base_config.copy()
            params[2*N] = theta_init
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective, 
                    params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    
            except Exception as e:
                pass
            
            # If we have time, do a second pass with tighter tolerance or different start
            if _time_left() > 20.0 and best_params is not None:
                # Perturb best solution slightly
                perturbed = best_params + np.random.randn(len(best_params)) * 0.1
                try:
                    res2 = minimize(
                        _objective, 
                        perturbed, 
                        method='L-BFGS-B', 
                        options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-9}
                    )
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_params = res2.x.copy()
                except:
                    pass

    if best_params is None:
        # Fallback
        inner = []
        for i in range(N):
            inner.append([i*2.0, 0.0, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        angle_deg = math.degrees(best_params[2*N])
        inner.append([x, y, angle_deg])
    
    center = (0.0, 0.0)
    side = _calculate_outer_side([(best_params[2*i], best_params[2*i+1]) for i in range(N)], best_params[2*N])
    angle_deg = math.degrees(best_params[2*N])
    
    return inner, center, side, angle_deg

