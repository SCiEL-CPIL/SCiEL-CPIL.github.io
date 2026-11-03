import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Lattice vectors for unit hexagons (side 1, touching)
    # Distance between centers is sqrt(3)
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    points = []
    # Generate a sufficient grid of points
    # Max coordinate needed is small for N=11
    for q in range(-5, 6):
        for r in range(-5, 6):
            p = q * v1 + r * v2
            points.append(p)
            
    # Calculate distances from origin
    dists = np.array([np.linalg.norm(p) for p in points])
    # Get indices of 11 closest points
    # Sort by distance
    idx = np.argsort(dists)
    selected_points = np.array([points[i] for i in idx[:N]])
    
    # Find min enclosing hexagon side S' for these points
    # The container hexagon has side S'.
    # Condition: max(p . n_k) <= S' * sqrt(3) / 2
    # So S' = 2/sqrt(3) * max(p . n_k)
    
    min_S_prime = float('inf')
    
    # The optimal rotation is likely aligned with the lattice points.
    # We can check angles 0 to 60 degrees.
    # Due to symmetry of the hexagon, we only need to check 0 to 60.
    # The "width" function is convex in log scale? No.
    # Let's sample angles.
    # The critical angles are where a point aligns with a normal or a vertex direction.
    # For small N, sampling is safe.
    
    angles = np.linspace(0, np.pi/3, 100) # 0 to 60 deg
    
    for theta in angles:
        # Normals at theta, theta+60, ...
        normals = []
        for k in range(6):
            angle = theta + k * np.pi/3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for p in selected_points:
            for n in normals:
                proj = np.dot(p, n)
                if proj > max_proj:
                    max_proj = proj
        
        S_prime = 2 * max_proj / np.sqrt(3)
        if S_prime < min_S_prime:
            min_S_prime = S_prime
            
    # The outer side length S
    S = min_S_prime + 1
    
    # The inner hexagons are at selected_points with angle 0
    inner = [[p[0], p[1], 0.0] for p in selected_points]
    
    # Outer hexagon center is (0,0), side is S, angle 0
    # Note: The angle of the outer hexagon in the return value should match the theta used?
    # The problem says "outer hexagon center (x, y), side length and rotation".
    # If we used theta for the normals, the rotation of the hexagon is theta (or theta+30 depending on def).
    # Let's assume the "rotation" parameter in the return value is the angle of the hexagon.
    # If normals are at theta, the sides are at theta+90.
    # Standard "angle" usually refers to the rotation of the shape.
    # If we return angle 0.0, the evaluator might assume a specific orientation.
    # The score is just the side length, but validity is checked.
    # We should return the angle that corresponds to the optimal packing we found.
    # If we found min_S_prime at theta_opt, we should return theta_opt as the rotation?
    # Or does the rotation parameter define the orientation of the container?
    # Yes. So we should return the theta that gave the min S'.
    
    # Let's re-run to find the best theta
    best_theta = 0
    min_S_prime_final = float('inf')
    for theta in angles:
        normals = []
        for k in range(6):
            angle = theta + k * np.pi/3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for p in selected_points:
            for n in normals:
                proj = np.dot(p, n)
                if proj > max_proj:
                    max_proj = proj
        
        S_prime = 2 * max_proj / np.sqrt(3)
        if S_prime < min_S_prime_final:
            min_S_prime_final = S_prime
            best_theta = theta

    S = min_S_prime_final + 1
    
    # Return
    # inner: list of [x, y, angle]
    # center: [x, y]
    # side: S
    # angle: best_theta (in degrees? The prompt says angle_degrees for inner, but for outer it just says "angle". 
    # Looking at the seed: `return inner, (0.0, 0.0), 40.0, 0.0`.
    # The inner has angle_degrees. The outer angle is likely in degrees too?
    # "outer hexagon center (x, y), side length and rotation"
    # Usually consistent units. Let's assume degrees.
    
    return inner, (0.0, 0.0), S, np.degrees(best_theta)
# EVOLVE-BLOCK-END
