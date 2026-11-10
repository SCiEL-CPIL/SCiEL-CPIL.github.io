import numpy as np

def hexagon_packing_11():
    # Centers of the 11 hexagons
    # 1 center + 6 neighbors + 4 from the second ring
    # Axial coordinates (q, r)
    # (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Second ring points to add: (2,0), (1,1), (0,2), (-1,2) ? 
    # Let's just pick 4 arbitrary from the second ring, symmetry might not matter for S if max proj is same.
    # Actually, to be robust, let's use a small optimization to find the best 4 points from the 2nd ring 
    # that minimize the max projection, although they are likely all equivalent.
    
    def axial_to_xy(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    centers = []
    # Center
    centers.append((0, 0))
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1:
        centers.append((q, r))
        
    # Ring 2
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2)]
    # We need 4 more. Let's pick (2,0), (1,1), (0,2), (-1,2) to be "upper" half? 
    # Or better, pick them symmetrically.
    # (2,0), (-2,0), (1,1), (-1,-1)? 
    # Let's just pick 4 distinct ones.
    # (2,0), (0,2), (-2,1), (0,-2)?
    # It doesn't matter for the bound S=1+2sqrt(3).
    # Let's pick (2,0), (1,1), (0,2), (-1,2).
    selected_ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2)]
    for q, r in selected_ring2:
        centers.append((q, r))
        
    xy_centers = [axial_to_xy(q, r) for q, r in centers]
    
    inner_hex_data = np.zeros((11, 3))
    for i, (x, y) in enumerate(xy_centers):
        inner_hex_data[i] = [x, y, 0]
        
    # Calculate S
    # Max projection of centers on normals
    # Normals for angle 0 outer hex: 30, 90, 150, 210, 270, 330
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append((np.cos(angle), np.sin(angle)))
        
    max_proj = 0
    for x, y in xy_centers:
        for nx, ny in normals:
            p = x * nx + y * ny
            if p > max_proj:
                max_proj = p
                
    # S * sqrt(3)/2 >= max_proj + 1
    S = (max_proj + 1) * 2 / np.sqrt(3)
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, S
