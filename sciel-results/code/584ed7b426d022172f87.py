import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal grid points
    # Axial coordinates (q, r)
    points_axial = []
    # Search radius large enough to get N points
    # For N=11, radius 2 is enough (1 + 6 + 12 = 19 points)
    max_radius = 5
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            # Axial distance
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            points_axial.append((dist, q, r))
    
    # Sort by distance from center
    points_axial.sort(key=lambda x: x[0])
    
    # Take first N points
    selected_axial = [p[1:] for p in points_axial[:N]]
    
    # Convert to Cartesian
    inner = []
    for q, r in selected_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Convert to numpy array for calculation
    inner_arr = np.array(inner)
    xs = inner_arr[:, 0]
    ys = inner_arr[:, 1]
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Try different rotations for the outer hexagon
    # The outer hexagon is regular. We check rotations from 0 to 30 degrees.
    # Due to symmetry, 0-30 is sufficient.
    for angle_deg in range(0, 31):
        angle_rad = np.deg2rad(angle_deg)
        # Normals for the 6 sides of the outer hexagon
        # The sides are at angles angle_rad + k * 60
        # The normal to a side is perpendicular to the side direction?
        # A regular hexagon centered at origin with "flat top" (vertices at 0, 60, ...) has sides with normals at 30, 90, ...
        # Let's define the hexagon by its side normals.
        # If we rotate the hexagon by `angle_rad`, the normals are at `angle_rad + k * 60`.
        # Wait, if we define the hexagon such that one side is horizontal (normal is vertical), 
        # and we rotate it, the normals rotate.
        
        # The condition for a point (x,y) to be inside a regular hexagon of side S centered at origin,
        # with normals at angles theta_k, is:
        # |x cos(theta_k) + y sin(theta_k)| <= S * sqrt(3) / 2
        
        # We want to find the minimum S such that this holds for all points and all k.
        # S >= (2 / sqrt(3)) * max_k ( max_i |x_i cos(theta_k) + y_i sin(theta_k)| )
        
        max_proj = 0.0
        for k in range(6):
            theta = angle_rad + k * np.pi / 3.0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            projections = np.abs(xs * cos_t + ys * sin_t)
            max_proj = max(max_proj, np.max(projections))
        
        required_side = (2.0 / np.sqrt(3.0)) * max_proj
        if required_side < best_side:
            best_side = required_side
            best_angle = angle_deg
            
    # Return in required format
    # inner: list of [x, y, angle]
    # center: (x, y)
    # side: float
    # angle: float (outer hexagon rotation)
    
    return inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
