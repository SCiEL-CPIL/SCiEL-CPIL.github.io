import numpy as np
import time
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill

    # 1. Define the 12-hexagon cluster topology
    # This is a compact cluster: 1 center, 6 in ring 1, 5 in ring 2.
    # Axial coordinates (q, r) for a hexagonal grid.
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: 
    #   (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)
    # We select 5 from Ring 2 to form a symmetric-ish shape.
    # Inspiration 1 used: (2,-1), (-2,1), (2,0), (-2,0), (1,-2), (-1,2)
    # Let's verify this set.
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This gives unit edge length hexagons with center distance sqrt(3).
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid

    # 2. Precompute unit hexagon vertices (angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # 3. Function to calculate minimal enclosing hexagon side length s
    # The outer hexagon has side length s and angle theta_outer.
    # We fix the outer hexagon angle to 0 for simplicity? 
    # No, we can rotate the inner cluster by theta relative to the outer hexagon.
    # This is equivalent to rotating the outer hexagon by -theta.
    # The problem allows specifying the outer hexagon angle.
    # Let's optimize the relative angle.
    
    def calc_s(theta):
        """
        Calculate the minimum side length s of a regular hexagon (angle 0)
        that contains all 12 hexagons rotated by theta.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        # Rotate centers
        rot_centers = centers @ rot.T
        
        # Rotate hexagon vertices (since the hexagons rotate with the cluster)
        rot_verts = hex_vertices @ rot.T
        
        # Generate all vertices of all hexagons
        # (12, 6, 2)
        all_verts = rot_centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 2)
        
        # The outer hexagon has angle 0.
        # Its vertices are at angles 0, 60, ...
        # Its edge normals are at 30, 90, 150, 210, 270, 330 degrees.
        # For a regular hexagon with side s and angle 0:
        # The distance from center to edge (apothem) is s * sqrt(3) / 2.
        # The projection of a point P onto a normal n is P . n.
        # The max projection must be <= apothem.
        # So s = max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3).
        
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto all normals
        # (72, 6)
        projs = all_verts @ normals.T
        
        # For each normal, find the max projection
        # We need the hexagon to contain all points, so for each normal direction,
        # the max projection determines the required apothem.
        max_projs = np.max(projs, axis=0)
        
        # The side length s is determined by the maximum required apothem
        # apothem = s * sqrt(3) / 2
        # s = 2 * apothem / sqrt(3)
        # We take the max over all 6 normals.
        max_apothem = np.max(max_projs)
        s_val = 2.0 * max_apothem / np.sqrt(3)
        
        return s_val

    # 4. Optimize theta
    # The cluster has 6-fold symmetry? 
    # Let's check the axial coords again.
    # (0,0)
    # (1,0), (-1,0) -> symmetric
    # (0,1), (0,-1) -> symmetric
    # (1,-1), (-1,1) -> symmetric
    # (2,-1), (-2,1) -> symmetric
    # (2,0), (-2,0) -> symmetric
    # (1,-2), (-1,2) -> symmetric
    # Yes, it has 6-fold rotational symmetry.
    # So we only need to search [0, 60 degrees).
    
    def objective(theta):
        return calc_s(theta)

    # Use minimize_scalar for 1D optimization
    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-10})
    
    if time.time() > deadline:
        # Fallback if optimization takes too long (unlikely)
        theta_opt = 0.0
    else:
        theta_opt = res.x
    
    s_opt = objective(theta_opt)
    
    # 5. Construct the result
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    rot = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(theta_opt) # Rotate the hexagons as well
        
    # Outer hexagon centered at 0,0.
    # Angle 0.
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"s: {s}")
