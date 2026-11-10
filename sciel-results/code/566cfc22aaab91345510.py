import numpy as np
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    # 1. Define the 12 inner hexagon centers in a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Convert axial to Cartesian
    centers = np.array([
        (np.sqrt(3) * (q + r/2), 1.5 * r) 
        for q, r in axial_coords
    ])
    
    # Scale to ensure separation > 1e-6
    # Original min distance is sqrt(3) ~ 1.732.
    # Hexagons touch at distance sqrt(3).
    # We need distance > sqrt(3) + epsilon.
    # Scaling by 1.001 increases distance by 0.1%.
    # 0.001 * 1.732 ~ 0.0017 > 1e-6. Safe.
    centers = centers * 1.001
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # 2. Generate vertices for each inner hexagon
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertex_offsets = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_vertices = []
    for c in centers:
        verts = c + vertex_offsets
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # 3. Optimization function
    def cost(theta):
        normal_angles = theta + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projections = all_vertices @ normals.T
        max_proj_per_vertex = np.max(projections, axis=1)
        A = np.max(max_proj_per_vertex)
        
        s = 2 * A / np.sqrt(3)
        return s

    result = minimize_scalar(cost, bounds=(0, np.pi/3), method='bounded')
    optimal_theta = result.x
    optimal_s = result.fun
    
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(optimal_theta)])
    
    return inner_hex_data, outer_hex_data, optimal_s
