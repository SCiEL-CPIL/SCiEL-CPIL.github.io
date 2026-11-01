<<<<<<< SEARCH
def analytic_upper_bound():
    """
    Construct a valid packing using a hexagonal lattice arrangement.
    12 hexagons in a hexagonal pattern: center + ring of 6 + partial second ring.
    For a hexagonal lattice of unit hexagons, the distance between centers of adjacent
    hexagons is sqrt(3) (the apothem*2 = side*sqrt(3) for unit hexagons).
    
    Let's place them in a hexagonal cluster. The minimal bounding hexagon for a
    hexagonal cluster of N hexagons can be computed.
    
    For N=12, a good arrangement is a 3x4 parallelogram or a hexagonal shape.
    Let's use a hexagonal lattice with basis vectors.
    """
    # Hexagonal lattice: place centers at positions that form a compact cluster.
    # For unit hexagons (side=1), the minimum distance between centers is sqrt(3).
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    a1 = (math.sqrt(3), 0)
    a2 = (math.sqrt(3)/2, 3/2)
    
    # Place 12 hexagons in a hexagonal pattern:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    positions = []
    # Let's use a simpler approach: 4 rows of 3
    # Or a hexagonal cluster
    # For now, use a 3x4 grid in hexagonal lattice coordinates
    
    # Hexagonal lattice positions for a compact shape:
    # (0,0), (1,0), (2,0), (3,0) - 4 in first row
    # (0,1), (1,1), (2,1), (3,1) - 4 in second row  
    # (0,2), (1,2), (2,2) - 3 in third row
    # (1,3) - 1 in fourth row
    # Total: 12
    
    lattice_points = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2),
        (1,3)
    ]
    
    inner = []
    for (i, j) in lattice_points:
        x = i * a1[0] + j * a2[0]
        y = i * a1[1] + j * a2[1]
        inner.append([x, y, 0.0])
    
    # Compute bounding hexagon
    # The outer hexagon needs to contain all these.
    # For now, estimate the side length by finding the extent.
    
    # Find the convex hull and fit a hexagon.
    # Simplified: find max distance from center in each of 6 directions.
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    # For a regular hexagon with flat-top orientation, the side length needed
    # to contain a set of points can be estimated.
    
    # Let's just use a large initial estimate and let the optimizer shrink it.
    # Compute a rough bound: max distance from center to any vertex of any inner hex.
    max_r = 0
    for xi, yi, ai in inner:
        vi = hex_vertices(xi, yi, 1.0, ai)
        for v in vi:
            r = math.sqrt((v[0]-cx)**2 + (v[1]-cy)**2)
            max_r = max(max_r, r)
    
    # For a regular hexagon, the circumradius = side, apothem = side*sqrt(3)/2
    # To contain a circle of radius max_r, we need side >= max_r (if aligned)
    # But the hexagon is not circular, so we need more.
    # Use side = max_r / (sqrt(3)/2) as a rough upper bound (apothem-based)
    side_est = max_r / (math.sqrt(3)/2) * 1.2  # add margin
    
    return inner, (cx, cy), side_est, 0.0
=======
def analytic_upper_bound():
    """
    Construct a valid packing using a hexagonal lattice arrangement.
    We use a 3x4 grid in hexagonal lattice coordinates to ensure non-overlap.
    """
    a1 = (math.sqrt(3), 0)
    a2 = (math.sqrt(3)/2, 1.5)
    
    # 3 rows of 4 hexagons
    lattice_points = []
    for j in range(3):
        for i in range(4):
            lattice_points.append((i, j))
            
    inner = []
    for (i, j) in lattice_points:
        x = i * a1[0] + j * a2[0]
        y = i * a1[1] + j * a2[1]
        inner.append([x, y, 0.0])
        
    # Center the cluster
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    # Shift all inner hexagons so their centroid is at (0,0)
    for p in inner:
        p[0] -= cx
        p[1] -= cy
        
    # Estimate outer hexagon side length
    # Find max extent in all 6 directions
    max_side = 0
    for xi, yi, ai in inner:
        vi = hex_vertices(xi, yi, 1.0, ai)
        for v in vi:
            # Project onto 6 normal vectors of a flat-top hexagon
            # Normals are at 0, 60, 120, 180, 240, 300 degrees
            for k in range(6):
                angle = k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(nx * v[0] + ny * v[1])
                # Apothem = side * sqrt(3)/2
                # side = proj / (sqrt(3)/2)
                side_needed = proj / (math.sqrt(3)/2)
                max_side = max(max_side, side_needed)
                
    # Add a small margin to ensure validity
    side_est = max_side + 0.1
    
    return inner, (0.0, 0.0), side_est, 0.0
>>>>>>> REPLACE

