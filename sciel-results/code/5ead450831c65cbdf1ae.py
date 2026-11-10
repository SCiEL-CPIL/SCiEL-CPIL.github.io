import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a lattice packing based on a 13-hexagon cluster with one corner removed.
    """
    # Lattice basis vectors for hexagonal packing
    # Distance between centers is sqrt(3)
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate centers for a 13-hexagon cluster (side length 2 in grid units)
    # Condition: |i| <= 2, |j| <= 2, |i+j| <= 2
    centers = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i) <= 2 and abs(j) <= 2 and abs(i+j) <= 2:
                center = i * a1 + j * a2
                centers.append(center)
                
    # We have 13 centers. Remove one to get 12.
    # Remove the one at (2, 0) in lattice coords, which is 2*a1
    # Find its index
    remove_idx = None
    target = 2 * a1
    for idx, c in enumerate(centers):
        if np.allclose(c, target):
            remove_idx = idx
            break
            
    if remove_idx is not None:
        centers.pop(remove_idx)
        
    # Now we have 12 centers
    # Convert to numpy array
    centers = np.array(centers)
    
    # Calculate required side length s for outer hexagon
    # Outer hexagon is centered at (0,0) with angle 0
    # Normals of the outer hexagon edges (pointing outwards)
    # For a hexagon with angle 0, vertices are at 0, 60, 120...
    # Edge 0 is between vertex 0 and 1? No.
    # Vertices: (s,0), (s/2, s*sqrt(3)/2)...
    # Edge 0: from (s/2, s*sqrt(3)/2) to (s, 0)? No, that's not an edge.
    # Let's check vertices for angle 0:
    # v0 = (s, 0)
    # v1 = (s/2, s*sqrt(3)/2)
    # Edge v0-v1 normal?
    # Vector v1-v0 = (-s/2, s*sqrt(3)/2). Normal is (s*sqrt(3)/2, s/2) -> angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    normals = []
    for k in range(6):
        angle = np.deg2rad(30 + 60 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    # Apothem of unit hexagon in these directions is sqrt(3)/2
    apothem = np.sqrt(3) / 2
    
    s = 0
    for n in normals:
        # Project all centers onto this normal
        projections = np.dot(centers, n)
        max_proj = np.max(projections)
        # The outer hexagon boundary is at distance s * cos(30) ? No.
        # The distance from center to the edge (apothem of outer hex) is s * sqrt(3)/2.
        # The condition is: max_projection + apothem_inner <= apothem_outer
        # apothem_outer = s * sqrt(3)/2
        # So s * sqrt(3)/2 >= max_proj + sqrt(3)/2
        # s >= (max_proj + sqrt(3)/2) / (sqrt(3)/2)
        # s >= 2 * (max_proj + sqrt(3)/2) / sqrt(3)
        # s >= 2 * max_proj / sqrt(3) + 1
        
        required_s = (max_proj + apothem) / (np.sqrt(3) / 2)
        if required_s > s:
            s = required_s
            
    # Prepare inner hexagon data
    # All inner hexagons have angle 0
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    # Prepare outer hexagon data
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
