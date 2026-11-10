import numpy as np
from itertools import combinations

def hexagon_packing_11():
    # Lattice vectors
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Core 7 hexagons
    core_indices = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    core_points = np.array([i*e1 + j*e2 for i,j in core_indices])
    
    # Second ring indices (12 points)
    ring2_indices = [
        (2,0), (-2,0),
        (1,1), (-1,-1),
        (0,2), (0,-2),
        (2,-1), (-2,1),
        (1,-2), (-1,2)
    ]
    ring2_points = np.array([i*e1 + j*e2 for i,j in ring2_indices])
    
    best_S = float('inf')
    best_config = None
    
    # Try all combinations of 4 points from ring2
    for combo in combinations(range(12), 4):
        added_points = ring2_points[list(combo)]
        all_points = np.vstack([core_points, added_points])
        
        # Find minimal enclosing hexagon side S
        # The optimal hexagon is likely concentric with the cluster's centroid
        # For symmetric clusters, centroid is (0,0)
        # We need to minimize S such that all points are inside the hexagon
        # Hexagon with center (0,0), side S, angle alpha
        # Vertices at S * exp(i * (alpha + k*60))
        # The condition for a point p to be inside is that its projection on each of the 6 edge normals is <= apothem
        # Apothem = S * sqrt(3)/2
        # Edge normals are at angles alpha + 30 + k*60
        
        # We can iterate over alpha
        # Due to symmetry of the lattice, we can restrict alpha
        min_S_for_alpha = float('inf')
        for alpha_deg in np.linspace(0, 60, 61): # 1 degree steps
            alpha = np.deg2rad(alpha_deg)
            normals = alpha + np.pi/6 + np.arange(6) * np.pi/3
            # Compute max projection of all_points onto each normal
            # p . n
            projections = all_points @ np.column_stack([np.cos(normals), np.sin(normals)])
            # The max absolute projection for each normal
            max_proj = np.max(np.abs(projections), axis=0)
            # The apothem must be at least max_proj
            apothem = np.max(max_proj)
            S = apothem * 2 / np.sqrt(3)
            if S < min_S_for_alpha:
                min_S_for_alpha = S
        
        if min_S_for_alpha < best_S:
            best_S = min_S_for_alpha
            best_config = all_points
            best_alpha = alpha_deg

    # Convert best_config to (x, y, angle)
    # All inner hexes have angle 0
    inner_hex_data = np.column_stack([best_config, np.zeros(11)])
    
    # Outer hexagon
    outer_hex_data = np.array([0, 0, best_alpha])
    
    return inner_hex_data, outer_hex_data, best_S
