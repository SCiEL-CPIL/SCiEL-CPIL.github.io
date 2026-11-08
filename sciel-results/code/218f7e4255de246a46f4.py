import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    N = 11
    
    # Define axial coordinates for 11 hexagons
    # Center + Ring 1 (6) + Ring 2 (4 selected to keep compact)
    # Spiral order or just list them
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Generate all vertices of all inner hexagons
    # Inner hexagons have side length 1.
    # Orientation: Let's assume vertices are at angles 0, 60, 120...
    # This orientation corresponds to a "pointy-top" hexagon? 
    # If vertices are at 0, 60, then the side between 0 and 60 is not horizontal/vertical.
    # It doesn't matter for the math as long as we calculate vertices correctly.
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * np.pi / 3.0
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            vertices.append((vx, vy))
    
    vertices = np.array(vertices)
    
    # Compute centroid of the vertices (approximate center of mass of the shape)
    center = np.mean(vertices, axis=0)
    
    # Function to calculate required side length S for a given outer rotation
    def calc_S(theta):
        # Normals for the 6 sides of the outer hexagon
        # The outer hexagon sides are perpendicular to these normals
        # If we define normals at theta, theta+pi/3...
        # The condition is dot(p - center, normal) <= apothem
        # apothem = S * sqrt(3) / 2
        # So S >= 2/sqrt(3) * dot(p - center, normal)
        
        max_proj = -np.inf
        for k in range(6):
            n_angle = theta + k * np.pi / 3.0
            normal = np.array([np.cos(n_angle), np.sin(n_angle)])
            # Project all vertices onto this normal
            # v - center
            diffs = vertices - center
            projs = diffs.dot(normal)
            max_proj = max(max_proj, np.max(projs))
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S
    
    # Sweep theta to find minimum S
    # The function is periodic with pi/3. We can search [0, pi/3]
    # Use a fine grid
    num_steps = 1000
    thetas = np.linspace(0, np.pi / 3.0, num_steps)
    min_S = np.inf
    best_theta = 0.0
    
    # Parallelization or fast loop
    # Since numpy is vectorized, we can calculate all at once?
    # Let's try vectorizing the projection calculation.
    
    # Normals matrix: (6, 2)
    # We can loop over thetas, it's fast enough (1000 iters)
    
    for i in range(num_steps):
        theta = thetas[i]
        S = calc_S(theta)
        if S < min_S:
            min_S = S
            best_theta = theta
        if time.time() > deadline:
            break
            
    # Refine the search around best_theta
    if time.time() < deadline - 5:
        # Local refinement
        step = (np.pi / 3.0) / num_steps
        fine_thetas = np.linspace(best_theta - step * 5, best_theta + step * 5, 50)
        for theta in fine_thetas:
            S = calc_S(theta)
            if S < min_S:
                min_S = S
                best_theta = theta
            if time.time() > deadline:
                break

    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for cx, cy in centers:
        # Angle 0.0 for all inner hexagons
        inner.append([cx, cy, 0.0])
        
    outer_center = [float(center[0]), float(center[1])]
    outer_side = float(min_S)
    outer_angle = float(best_theta * 180.0 / np.pi)
    
    return inner, outer_center, outer_side, outer_angle
