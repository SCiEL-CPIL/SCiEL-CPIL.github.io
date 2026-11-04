<<<<<<< SEARCH
    d = np.sqrt(3)
    
    # 11-hexagonal rigid cluster: 7-hex cluster + 4 extensions
    # All hexagons have the same orientation (0 degrees relative to cluster)
    cluster_centers = np.array([
        [0, 0],
        [d, 0], [d/2, 1.5], [-d/2, 1.5],
        [-d, 0], [-d/2, -1.5], [d/2, -1.5],
        [2*d, 0],
        [d, 3.0],
        [-d, 3.0],
        [-2*d, 0]
    ])
=======
    d = np.sqrt(3)
    
    # 11-hexagonal rigid cluster based on triangular lattice
    # Lattice vectors for unit hexagons (side 1, vertex at 0 deg):
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    
    # Lattice coordinates (i, j) for the 11 closest points to origin
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,-1), (2,-1), (-2,1)
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    
    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
>>>>>>> REPLACE
