import numpy as np
from scipy.optimize import minimize_scalar, minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 630.0  # Evaluator kills at 630s, keep safety margin
    
    # 12 unit hexagons packed in a rigid cluster.
    # Axial coordinates (q, r) for a hexagonal grid.
    # Adjacent hexagons in this grid have center-to-center distance sqrt(3).
    # We use a compact shape: a hexagon of side 2 (19 cells) minus 7 specific cells.
    # Let's use the shape from the seed which was valid:
    # (0, 0)
    # (1, 0), (-1, 0)
    # (0, 1), (0, -1)
    # (1, -1), (-1, 1)
    # (2, -1), (-2, 1)
    # (2, 0), (-2, 0)
    # (1, -2), (-1, 2)
    
    axial_coords = np.array([
        [0, 0],
        [1, 0], [-1, 0],
        [0, 1], [0, -1],
        [1, -1], [-1, 1],
        [2, -1], [-2, 1],
        [2, 0], [-2, 0],
        [1, -2], [-1, 2]
    ], dtype=float)
    
    # Convert axial (q, r) to Cartesian (x, y) for unit hexagons (side 1).
    # Standard mapping:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Distance between (0,0) and (1,0) is sqrt(3). Correct.
    # Distance between (0,0) and (0,1) is 1.5. Wait.
    # Let's check distance between (0,0) and (0,1).
    # x1=0, y1=0. x2=sqrt(3)*0.5, y2=1.5. dist = sqrt(3/4 + 2.25) = sqrt(3). Correct.
    
    centers = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial_coords):
        centers[i, 0] = np.sqrt(3) * (q + r / 2.0)
        centers[i, 1] = 1.5 * r
        
    # Center the cluster at the origin to simplify rotation and translation
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Precompute vertices of a unit hexagon at origin, angle 0.
    # Vertices at angles 0, 60, ..., 300.
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # The outer hexagon will be centered at the origin (we can shift the whole system later if needed, 
    # but since we minimize the bounding box, centering the inner cluster at origin and outer at origin is optimal 
    # for a symmetric bounding shape if we allow translation. 
    # Actually, the optimal translation of the inner cluster relative to the outer hexagon center is 0 
    # if the cluster is symmetric? Not necessarily. 
    # However, the bounding box of a set of points is minimized when the "center of the bounding box" 
    # aligns with the "center of the shape". 
    # Since the outer hexagon is regular and centered at origin, its support function is h(u) = s * ||u||_hex.
    # The minimal s is max over all vertices v of the inner cluster of ||v||_hex.
    # We can translate the inner cluster by t to minimize max_v ||v + t||_hex.
    # Due to symmetry of the hex norm and the cluster, t=0 is often optimal, but let's check.
    # For now, we fix t=0 (inner cluster centered at origin) and optimize rotation.
    # If we want to be safe, we could optimize t as well, but that adds 2 variables.
    # Given the 6-fold symmetry of the hex norm, and the near-symmetry of the cluster, t=0 is a strong candidate.
    
    # Function to compute the minimal side length s of the outer hexagon containing the rotated cluster.
    # The outer hexagon has side s, centered at origin, angle 0.
    # Its vertices are at s * (cos(60k), sin(60k)).
    # The condition for a point p to be inside is:
    # max over 6 normals n_k of dot(p, n_k) <= s * sqrt(3) / 2
    # The normals are at angles 30, 90, 150, 210, 270, 330.
    
    normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def calc_s(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        # Rotate centers
        rot_centers = centers @ rot.T
        
        # The inner hexagons are also rotated by theta.
        # Vertices of inner hexagon i: rot_centers[i] + rot @ hex_vertices.T
        # We need the max projection of ALL vertices of ALL inner hexagons onto the 6 normals.
        
        # Instead of looping over 12 hexagons and 6 vertices, we can vectorize.
        # All vertices = rot_centers[:, np.newaxis, :] + (rot @ hex_vertices.T)[np.newaxis, :, :]
        # This is (12, 6, 2).
        
        # Project onto normals (6, 2).
        # projs = vertices @ normals.T -> (12, 6, 6)
        # max over last two axes.
        
        # To save memory and time, we can just loop over the 12 hexagons, it's fast enough.
        max_proj = 0.0
        for i in range(12):
            verts = rot_centers[i] + (hex_vertices @ rot.T)
            projs = verts @ normals.T
            hex_max = np.max(projs)
            if hex_max > max_proj:
                max_proj = hex_max
                
        s_val = max_proj * 2.0 / np.sqrt(3)
        return s_val
    
    # Optimize theta
    # The cluster has 6-fold symmetry?
    # Let's check the axial coords again.
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,-1), (-1,1)
    # (2,-1), (-2,1)
    # (2,0), (-2,0)
    # (1,-2), (-1,2)
    # Rotation by 60 degrees in axial coords: (q, r) -> (-r, q+r)
    # (0,0) -> (0,0)
    # (1,0) -> (0,1). Is (0,1) in the list? Yes.
    # (0,1) -> (-1,1). Is (-1,1) in the list? Yes.
    # (-1,1) -> (-1,0). Is (-1,0) in the list? Yes.
    # (-1,0) -> (0,-1). Is (0,-1) in the list? Yes.
    # (0,-1) -> (1,-1). Is (1,-1) in the list? Yes.
    # (1,-1) -> (1,0). Is (1,0) in the list? Yes.
    # (2,-1) -> (1,1). Is (1,1) in the list? NO.
    # So it does NOT have 6-fold symmetry.
    # Let's check 120 degrees: (q,r) -> (-q-r, q)
    # (2,-1) -> (-1, 2). Is (-1,2) in the list? Yes.
    # (-2,1) -> (1, -2). Is (1,-2) in the list? Yes.
    # (2,0) -> (-2, 2). Is (-2,2) in the list? NO.
    # So it has 3-fold symmetry?
    # Let's check (2,0) -> (-2,2). Not in list.
    # So no rotational symmetry?
    # Wait, let's re-verify the list.
    # The list is:
    # 0: (0,0)
    # 1: (1,0)
    # 2: (-1,0)
    # 3: (0,1)
    # 4: (0,-1)
    # 5: (1,-1)
    # 6: (-1,1)
    # 7: (2,-1)
    # 8: (-2,1)
    # 9: (2,0)
    # 10: (-2,0)
    # 11: (1,-2)
    # 12: (-1,2) -> Wait, I only listed 12 items in the array above.
    # My array:
    # [0, 0],
    # [1, 0], [-1, 0],
    # [0, 1], [0, -1],
    # [1, -1], [-1, 1],
    # [2, -1], [-2, 1],
    # [2, 0], [-2, 0],
    # [1, -2], [-1, 2]
    # Total 12.
    
    # Let's check symmetry of this specific set.
    # Rotation 60: (q,r)->(-r, q+r)
    # (2,0) -> (0,2). Not in set.
    # So no 60 deg symmetry.
    # We must search full 360 degrees.
    
    def objective(theta):
        return calc_s(theta)
    
    # Use a coarse grid to find a good initial guess, then refine.
    thetas = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    s_vals = np.array([calc_s(t) for t in thetas])
    idx_min = np.argmin(s_vals)
    theta_init = thetas[idx_min]
    
    # Refine around the best grid point
    # Search in [theta_init - pi/100, theta_init + pi/100]
    bounds = (theta_init - np.pi/100, theta_init + np.pi/100)
    # Ensure bounds are within [0, 2pi] for the optimizer if needed, but bounded method handles any range.
    
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-10})
    
    # Check time
    if time.time() > deadline - 10:
        # Use the grid result if optimization took too long
        theta_opt = theta_init
        s_opt = s_vals[idx_min]
    else:
        theta_opt = res.x
        s_opt = objective(theta_opt)
        
    # Construct result
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    rot = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(theta_opt)
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt
