# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    # 1. Define 12 unit hexagon centers in a compact cluster
    # Axial coordinates (q, r)
    # This is a "flower" (7 hexes) + 5 more on the perimeter
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1)
    ]
    
    # Scale by a small factor to ensure separation (gap > 1e-6)
    # Distance between adjacent centers is sqrt(3).
    # Scaling by (1+delta) creates a gap of sqrt(3)*delta.
    # We need sqrt(3)*delta > 1e-6 => delta > ~6e-7.
    # Use 1e-4 for safety.
    scale = 1.0 + 1e-4
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q * scale
        y = np.sqrt(3) * (r + 0.5 * q) * scale
        centers.append([x, y])
    centers = np.array(centers)
    
    # Generate vertices for all 12 hexagons
    # Hexagon vertices relative to center, side 1, angle 0
    angles = np.deg2rad(np.arange(0, 360, 60))
    rel_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_vertices = []
    for c in centers:
        verts = c + rel_vertices
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices) # Shape (72, 2)
    
    # 2. Define the objective function to minimize
    # Outer hexagon normals (perpendicular to edges)
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # Shape (6, 2)
    
    def objective(T):
        tx, ty = T
        shifted_vertices = all_vertices + np.array([tx, ty])
        # Project onto all 6 normals
        projections = shifted_vertices @ normals.T
        max_abs_proj = np.max(np.abs(projections))
        s = (2.0 / np.sqrt(3)) * max_abs_proj
        return s

    # 3. Optimize translation T
    initial_T = -np.mean(all_vertices, axis=0)
    
    # Use Nelder-Mead for small dimension optimization
    res = minimize(objective, initial_T, method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000})
    
    optimal_T = res.x
    optimal_s = res.fun
    
    # 4. Construct the result
    final_centers = centers + optimal_T
    
    # Inner hexagons: (x, y, angle_deg)
    inner_hex_data = np.column_stack([final_centers, np.zeros(12)])
    
    # Outer hexagon: centered at (0,0), angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, optimal_s
# EVOLVE-BLOCK-END
