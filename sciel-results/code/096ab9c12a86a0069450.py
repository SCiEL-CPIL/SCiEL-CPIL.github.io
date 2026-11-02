def _make_hex_cluster():
    """Create 11 hexagons in a compact cluster.
    Uses a 7-hex core (side 2 hexagon) plus 4 hexes at cardinal directions.
    Axial coordinates (i, j) -> Cartesian:
    x = sqrt(3) * (i + j/2)
    y = 1.5 * j
    """
    sqrt3 = math.sqrt(3)
    angle = 30.0
    
    # Axial coordinates for the 11 hexagons
    # Core 7:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extra 4:
    # (2,0), (0,2), (-2,0), (0,-2)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    positions = []
    for i, j in axial_coords:
        x = sqrt3 * (i + j / 2.0)
        y = 1.5 * j
        positions.append((x, y))
        
    # Center the cluster
    cx_avg = sum(p[0] for p in positions) / len(positions)
    cy_avg = sum(p[1] for p in positions) / len(positions)
    
    inner = []
    for px, py in positions:
        inner.append([px - cx_avg, py - cy_avg, angle])
        
    return inner
