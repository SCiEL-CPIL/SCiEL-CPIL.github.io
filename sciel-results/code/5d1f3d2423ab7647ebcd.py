def _tight_lattice_init():
    # Generate a grid of hexagonal lattice points
    # Distance between centers is sqrt(3)
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5]) # Note: 3/2 = 1.5
    
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            pts.append(p)
    pts = np.array(pts)
    
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx]
    
    # Take 12 closest
    return pts[:12]
