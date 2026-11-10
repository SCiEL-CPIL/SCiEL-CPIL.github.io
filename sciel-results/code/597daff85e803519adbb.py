import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Generate initial positions from the hexagonal lattice
    # Lattice vectors for touching hexagons (side 1)
    # Distance between centers is sqrt(3)
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3/2, 1.5])
    
    # Generate a grid of lattice points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            points.append(p)
    points = np.array(points)
    
    # Select the 11 points closest to the origin
    distances = np.linalg.norm(points, axis=1)
    indices = np.argsort(distances)[:n]
    init_centers = points[indices]
    
    # Ensure we have 11 points. If duplicates or issues, fallback to manual
    if len(init_centers) < 11:
         # Fallback to simple grid
         init_centers = np.array([
             [0, 0],
             [-sqrt3, 0], [sqrt3, 0],
             [-sqrt3/2, 1.5], [sqrt3/2, 1.5],
             [-sqrt3/2, -1.5], [sqrt3/2, -1.5],
             [-1.5*sqrt3, 1.5], [1.5*sqrt3, 1.5],
             [-1.5*sqrt3, -1.5], [1.5*sqrt3, -1.5]
         ])

    # Define the objective function
    def objective(x):
        # x is a flattened array of 22 coordinates (11 * 2)
        centers = x.reshape((n, 2))
        
        # Calculate vertices for all hexagons
        # Hexagon vertices relative to center:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Wait, the prompt says vertices are (cx + s*cos(a + 60i), ...).
        # With s=1, a=0:
        # i=0: (1, 0)
        # i=1: (0.5, sqrt(3)/2)
        # i=2: (-0.5, sqrt(3)/2)
        # i=3: (-1, 0)
        # i=4: (-0.5, -sqrt(3)/2)
        # i=5: (0.5, -sqrt(3)/2)
        verts_rel = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        # All vertices in absolute coordinates
        # Shape (11, 6, 2)
        all_verts = centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]
        
        # Calculate required apothem A_req
        # The container is a regular hexagon. We can assume its orientation is fixed
        # (e.g., flat top/bottom) because we can rotate the cluster if needed, 
        # but optimizing positions in a fixed frame is sufficient if we allow 
        # the cluster to be any shape. 
        # The minimal S for a given set of points is determined by the max projection
        # onto the 6 normals of the container.
        # Container normals (for flat-top hexagon, side S, apothem A):
        # Normals point to the midpoints of the edges.
        # Angles: 0, 60, 120, 180, 240, 300 degrees?
        # If vertices are at 0, 60..., edges are at 30, 90...
        # Let's use normals at 0, 60, 120, 180, 240, 300.
        # This corresponds to a "pointy-top" container?
        # Let's check the vertex definition in prompt:
        # "vertices (cx + s*cos(a + 60i deg)..."
        # If a=0, vertex at 0 deg (right).
        # This is a "pointy-right" hexagon.
        # The edges are perpendicular to angles 30, 90, 150, 210, 270, 330.
        # So the normals of the container (if aligned with inner hexagons) would be at 30, 90, etc.
        # However, we can rotate the container. The minimal S is the same regardless of rotation 
        # if we consider all 6 directions.
        # The projection of a point (x,y) onto a normal at angle theta is x*cos(theta) + y*sin(theta).
        # We need max over all points and all 6 normal directions.
        
        normal_angles = np.deg2rad(np.arange(0, 360, 60)) # 0, 60, ...
        # Wait, if the inner hexagons are aligned (a=0), their "flat" sides are at 30, 90, etc.
        # If we align the container's flat sides with the inner hexagons' flat sides, 
        # the container normals should be at 30, 90, 150, 210, 270, 330.
        # Let's use these normals.
        container_normal_angles = np.deg2rad(np.arange(30, 390, 60))
        cos_n = np.cos(container_normal_angles)
        sin_n = np.sin(container_normal_angles)
        
        # Project all vertices onto these normals
        # all_verts: (11, 6, 2)
        # cos_n: (6,)
        # We want max projection for each normal, then max over normals.
        
        # Flatten vertices
        verts_flat = all_verts.reshape(-1, 2)
        
        # Calculate projections
        # proj[k, i] = v_i . n_k
        proj = verts_flat @ np.stack([cos_n, sin_n], axis=1).T
        # proj shape: (66, 6)
        
        # Max projection for each normal
        max_proj_per_normal = np.max(proj, axis=0)
        
        # Required apothem is the max of these
        A_req = np.max(max_proj_per_normal)
        
        # Side length S = 2 * A / sqrt(3)
        S = 2 * A_req / sqrt3
        
        # Penalty for overlaps
        # We assume inner hexagons are aligned (angle 0).
        # Overlap if center distance < sqrt(3).
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < sqrt3:
                    # Quadratic penalty for violation
                    penalty += (sqrt3 - dist)**2 * 1000
        
        return S + penalty

    # Initial guess
    x0 = init_centers.flatten()
    
    # Optimize
    # Use Nelder-Mead for robustness without gradients
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Extract optimized centers
    opt_centers = res.x.reshape((n, 2))
    
    # Calculate final S
    # Recalculate A_req and S for the final positions
    sqrt3 = np.sqrt(3)
    verts_rel = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    all_verts = opt_centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]
    verts_flat = all_verts.reshape(-1, 2)
    container_normal_angles = np.deg2rad(np.arange(30, 390, 60))
    cos_n = np.cos(container_normal_angles)
    sin_n = np.sin(container_normal_angles)
    proj = verts_flat @ np.stack([cos_n, sin_n], axis=1).T
    max_proj_per_normal = np.max(proj, axis=0)
    A_req = np.max(max_proj_per_normal)
    S_final = 2 * A_req / sqrt3
    
    # Construct return data
    # Inner hexagons: (x, y, angle_deg). Angle is 0 for all.
    inner_hex_data = np.column_stack((opt_centers[:, 0], opt_centers[:, 1], np.zeros(n)))
    
    # Outer hexagon: centered at (0,0), angle 0.
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(S_final)
