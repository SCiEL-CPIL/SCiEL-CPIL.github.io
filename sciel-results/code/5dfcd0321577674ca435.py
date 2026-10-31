import numpy as np
import math
import time

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    """
    Get vertices of a regular hexagon.
    Side length 'side'.
    Angle is rotation in degrees.
    """
    angle_rad = math.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0 + angle_rad
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def generate_lattice_points(n):
    """
    Generate n points in a hexagonal lattice, centered around (0,0).
    Distance between adjacent centers is sqrt(3) to ensure unit hexagons (side 1) are disjoint.
    """
    d = math.sqrt(3.0)
    # Lattice vectors
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3.0)/2.0])
    
    points = []
    # Generate a grid of lattice coordinates
    # We need enough range to get N points
    max_coord = 5
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            p = i * u + j * v
            points.append((i*i + j*j + i*j, p)) # Use a distance metric for sorting
            
    # Sort by distance from origin
    points.sort(key=lambda x: x[0])
    
    # Take first N
    return [p for _, p in points[:n]]

def min_enclosing_hexagon_side(points, angle_rad):
    """
    Calculate the minimum side length of a regular hexagon (centered at 0,0)
    that contains all points in 'points', given the relative rotation angle_rad.
    The outer hexagon is aligned with the axes (normals at 30, 90, ... degrees).
    The points are rotated by angle_rad relative to this frame.
    """
    if len(points) == 0:
        return 0.0
    
    pts = np.array(points)
    # Rotate points by -angle_rad so we can test against a fixed hexagon?
    # Or rotate normals?
    # Let's rotate points by angle_rad.
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot_pts = pts @ np.array([[c, -s], [s, c]]).T
    
    # Normals of the outer hexagon (apothem directions)
    # Vertices at 0, 60... -> Normals at 30, 90...
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project rotated points onto normals
    proj = rot_pts @ normals.T
    
    # Max absolute projection for each normal
    max_abs_proj = np.max(np.abs(proj), axis=0)
    
    # Required apothem
    required_apothem = np.max(max_abs_proj)
    
    # Side length S = A / (sqrt(3)/2)
    S = required_apothem * 2.0 / math.sqrt(3.0)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate the fixed cluster of centers
    centers = generate_lattice_points(N)
    
    # 2. Optimize the rotation angle to minimize the enclosing hexagon side
    # The function is periodic with period 60 degrees (pi/3).
    # We can scan or use a simple minimizer.
    
    best_angle = 0.0
    best_side = float('inf')
    
    # Scan the angle from 0 to 60 degrees
    # Use a fine grid first, then refine?
    # Or just use scipy.optimize.minimize_scalar on [0, pi/3]
    
    import scipy.optimize as opt
    
    def objective(angle):
        return min_enclosing_hexagon_side(centers, angle)
        
    try:
        res = opt.minimize_scalar(objective, bounds=(0, math.pi/3), method='bounded', 
                                  options={'xatol': 1e-6})
        if res.success:
            best_angle = res.x
            best_side = res.fun
    except Exception:
        # Fallback to grid search
        angles = np.linspace(0, math.pi/3, 100)
        for a in angles:
            s = min_enclosing_hexagon_side(centers, a)
            if s < best_side:
                best_side = s
                best_angle = a
                
    # 3. Construct the final configuration
    # The outer hexagon is centered at (0,0) with angle 0.
    # The inner hexagons are centered at the rotated centers.
    # The inner hexagons have angle 0? 
    # Wait, if we rotate the centers, do we rotate the hexagons themselves?
    # Yes, to maintain the lattice structure (disjointness), the hexagons should rotate with the centers.
    # So inner hexagon i is at rotated_center_i with rotation angle best_angle.
    
    c = math.cos(best_angle)
    s = math.sin(best_angle)
    
    inner = []
    for p in centers:
        x, y = p
        # Rotate center
        rx = x * c - y * s
        ry = x * s + y * c
        inner.append([rx, ry, math.degrees(best_angle)])
        
    # Outer hexagon: center (0,0), side best_side, angle 0
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

