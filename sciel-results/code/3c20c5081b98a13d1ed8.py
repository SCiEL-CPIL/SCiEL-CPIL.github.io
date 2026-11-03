<<<<<<< SEARCH
def hexagonal_lattice_config():
    """Generate a hexagonal lattice arrangement of 11 hexagons."""
    # Flat-top hexagons: neighbors at angles 0, 60, 120, 180, 240, 300 with distance sqrt(3)
    d = np.sqrt(3)
    centers = [(0.0, 0.0)]
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    # Second ring: need 4 more to make 11
    # Positions at distance 2d in directions 0, 60, 120, 180, 240, 300
    # And positions at distance d*sqrt(3) in directions 30, 90, 150, 210, 270, 330
    # Let's add 4 from the second ring
    added = 0
    for k in range(6):
        if added >= 4:
            break
        angle = np.radians(60 * k)
        cx = 2 * d * np.cos(angle)
        cy = 2 * d * np.sin(angle)
        centers.append((cx, cy))
        added += 1
    
    inner = [[x, y, 0.0] for x, y in centers[:11]]
    return inner
=======
def hexagonal_lattice_config():
    """Generate a hexagonal lattice arrangement of 11 hexagons."""
    # Flat-top hexagons (rotation 0): vertices at 0, 60, 120, 180, 240, 300.
    # Neighbors share edges. The normals to the edges are at 30, 90, 150, 210, 270, 330.
    # Distance between centers of touching hexagons is sqrt(3).
    d = np.sqrt(3)
    centers = [(0.0, 0.0)]
    # First ring: 6 hexagons at angles 30, 90, 150, 210, 270, 330
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    # Second ring: need 4 more to make 11
    # The second ring positions in a full hexagonal cluster are at distance 2*d in the 6 directions of the first ring,
    # and at distance d*sqrt(3) in the 6 directions of the vertices (0, 60, ...).
    # Let's add 4 from the 'corner' positions of the second ring (distance 2d) to keep it compact.
    added = 0
    for k in range(6):
        if added >= 4:
            break
        angle = np.radians(30 + 60 * k)
        cx = 2 * d * np.cos(angle)
        cy = 2 * d * np.sin(angle)
        centers.append((cx, cy))
        added += 1
    
    inner = [[x, y, 0.0] for x, y in centers[:11]]
    return inner
>>>>>>> REPLACE

