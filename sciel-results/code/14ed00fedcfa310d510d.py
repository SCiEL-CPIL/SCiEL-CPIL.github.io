<<<<<<< SEARCH
    # K-points in the irreducible BZ
    # Gamma (0,0) to X (0.5, 0) to M (0.5, 0.5)
    # We sample points along these lines.
    k_points = []
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0))
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5, t))
    # Add Gamma explicitly if not covered well
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    
    # Unique k points
    k_points = list(set([(round(k[0], 3), round(k[1], 3)) for k in k_points]))
=======
    # K-points in the irreducible BZ
    # Gamma (0,0) to X (0.5, 0) to M (0.5, 0.5)
    # We sample points along these lines.
    k_points = []
    # Add more points for better accuracy of the gap
    num_k = 10
    for t in np.linspace(0, 0.5, num_k):
        k_points.append((t, 0))
    for t in np.linspace(0, 0.5, num_k):
        k_points.append((0.5, t))
    
    # Unique k points
    k_points = list(set([(round(k[0], 4), round(k[1], 4)) for k in k_points]))
>>>>>>> REPLACE

