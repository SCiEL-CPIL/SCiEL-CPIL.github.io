<<<<<<< SEARCH
    # Lattice basis vectors for unit hexagons touching side-to-side
    # u1 is along x-axis, u2 is at 60 degrees
    # Distance between centers is sqrt(3)
    u1 = np.array([np.sqrt(3), 0])
    u2 = np.array([np.sqrt(3)/2, 1.5])
=======
    # Lattice basis vectors for unit hexagons
    # Slightly larger than sqrt(3) to ensure strict disjointness (no touching)
    # u1 is along x-axis, u2 is at 60 degrees
    scale = 1.01
    u1 = np.array([np.sqrt(3) * scale, 0])
    u2 = np.array([np.sqrt(3)/2 * scale, 1.5 * scale])
>>>>>>> REPLACE
