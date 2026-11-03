import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Axial coordinates for a symmetric cluster of 11 hexagons
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 subset (symmetric): (1,1), (-1,2), (-1,-1), (1,-2)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-1, -1), (1, -2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Vertices of a unit regular hexagon (pointy top)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def calc_side_length(theta_rad):
        # Outer hexagon normals
        # If outer hexagon is rotated by theta, its sides are perpendicular to normals at theta, theta+60...
        # Let's define normals at theta + k*60
        outer_normals = []
        for k in range(6):
            angle = theta_rad + k * np.pi / 3.0
            outer_normals.append(np.array([np.cos(angle), np.sin(angle)]))
        outer_normals = np.array(outer_normals)
        
        max_dist = 0.0
        for n in outer_normals:
            # R_inner(n) is max projection of unit hexagon vertices onto n
            projections = vertices @ n
            r_inner = np.max(projections)
            
            # Distance from outer center to the face of the inner hexagon i in direction n
            # dist = c_i . n + r_inner
            dists = centers @ n + r_inner
            current_max = np.max(dists)
            if current_max > max_dist:
                max_dist = current_max
                
        # S * sqrt(3)/2 = max_dist
        # S = max_dist * 2 / sqrt(3)
        side_length = max_dist * 2.0 / np.sqrt(3)
        return side_length

    # Optimize theta
    best_theta = 0.0
    best_side = float('inf')
    
    # Sample theta from 0 to 60 degrees
    # The system has 6-fold symmetry, so 0 to 60 is sufficient
    thetas = np.linspace(0, np.pi/3, 601) # 0.1 degree steps
    
    for th in thetas:
        s = calc_side_length(th)
        if s < best_side:
            best_side = s
            best_theta = th
            
    # Construct inner hexagons list
    inner = []
    for x, y in centers:
        # Angle is 0 for all
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon center is (0,0)
    # Side length is best_side
    # Rotation is best_theta (in degrees)
    # Note: The interface asks for (x, y, side, angle). 
    # "outer hexagon center (x, y), side length and rotation"
    # So return inner, center, side, angle
    
    return inner, (0.0, 0.0), float(best_side), float(np.rad2deg(best_theta))
# EVOLVE-BLOCK-END
