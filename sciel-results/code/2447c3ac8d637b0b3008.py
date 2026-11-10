<<<<<<< SEARCH
    # Calculate centers
    centers = np.array([i * a1 + j * a2 for i, j in coords])
    
    # Unit hexagon vertices (angle 0, centered at origin)
    hex_angles = np.arange(6) * np.pi / 3.0
    unit_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])  # (6,2)

    # Generate all vertices of the 12 hexagons
    # Centers shape (12, 2), unit_verts shape (6, 2)
    all_verts = (centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]).reshape(-1, 2)
=======
    # Calculate centers
    centers = np.array([i * a1 + j * a2 for i, j in coords])
    
    # Center the cluster at the origin to potentially minimize the bounding hexagon
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Scale the cluster slightly to ensure a gap > 1e-6 between touching hexagons
    # This prevents "intersection" errors due to zero gap
    scale_factor = 1.0 + 1e-5
    centers = centers * scale_factor

    # Unit hexagon vertices (angle 0, centered at origin)
    hex_angles = np.arange(6) * np.pi / 3.0
    unit_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])  # (6,2)

    # Generate all vertices of the 12 hexagons
    # Centers shape (12, 2), unit_verts shape (6, 2)
    all_verts = (centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]).reshape(-1, 2)
>>>>>>> REPLACE
