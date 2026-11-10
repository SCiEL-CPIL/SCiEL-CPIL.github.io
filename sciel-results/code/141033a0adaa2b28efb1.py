"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega, omega = exp(i*pi/3), which are at
    mutual distance at least one. Maximizes the number of pairs at distance
    exactly one; Harborth's theorem gives the optimum floor(3n - sqrt(12n - 3)) = 76.
    
    We use a deterministic approach: generate a triangular lattice spiral of
    32 points that achieves the Harborth bound of 76 edges.
    
    The triangular lattice in axial coordinates (a, b) has 6 neighbors:
    (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    A hexagonal spiral (or similar compact shape) of 32 vertices on the
    triangular lattice should achieve 76 edges per Harborth's theorem.
    
    Let's construct a compact cluster. The optimal shape for maximizing
    edges is a "hexagonal" or "triangular" cluster. For n=32, we can
    build a hexagon with side lengths that fit 32 points.
    
    A hexagon with side length k has 3k(k-1)+1 vertices.
    k=3: 3*3*2+1 = 19
    k=4: 3*4*3+1 = 37
    
    So a hexagon of side 3 has 19 vertices, and we need 13 more.
    
    Alternatively, let's just use a known construction. The triangular
    lattice spiral (filling rings) gives:
    Ring 0: 1 point (center)
    Ring 1: 6 points (total 7)
    Ring 2: 12 points (total 19)
    Ring 3: 18 points (total 37)
    
    We need 32 points, so we take rings 0, 1, 2 (19 points) plus 13 from ring 3.
    
    Let me generate the hexagonal spiral points and take the first 32.
    """
    points = []
    
    # Generate hexagonal spiral points in order of distance from origin
    # In axial coordinates, the "distance" from origin for hexagonal shapes
    # is max(|a|, |b|, |a+b|) ... actually for triangular lattice, let's use
    # the hexagonal distance: max(|a|, |b|, |a+b|)
    
    # Actually, let me think more carefully. The triangular lattice point (a,b)
    # has 6 neighbors. A compact cluster should be roughly hexagonal.
    
    # Let's generate points by their "ring" number (hexagonal distance)
    # and fill them in order.
    
    # Hexagonal distance in axial coords: d = max(|a|, |b|, |a+b|)
    # Wait, that's not quite right for triangular lattice. Let me reconsider.
    
    # For the triangular lattice with basis vectors e1=(1,0) and e2=(1/2, sqrt(3)/2),
    # the point (a,b) in axial coords corresponds to Cartesian (a + b/2, b*sqrt(3)/2).
    # The 6 neighbors are at: (a±1,b), (a,b±1), (a+1,b-1), (a-1,b+1)
    
    # A good measure of "distance from center" for building compact shapes:
    # Use the hexagonal distance: max(|a|, |b|, |a+b|)
    
    # Let's collect all points with |a|,|b| <= 64, sort by hex distance, then by angle
    # to get a spiral-like ordering, and take the first 32.
    
    # Actually, simpler: just enumerate points in a hexagonal spiral pattern.
    
    # Hex spiral generation:
    # Ring 0: (0,0)
    # Ring r has 6*r points, located at the boundary of the hexagon of "radius" r.
    
    # For ring r, the 6 sides are:
    # Side 1: (r, 0) to (r, -r) -> a=r, b from 0 to -r
    # Side 2: (r, -r) to (0, -2r) -> a from r to 0, b = -a - r ... hmm
    
    # Let me use a standard hex spiral algorithm.
    # Direction vectors for the 6 sides:
    # d0 = (1, 0), d1 = (0, -1), d2 = (-1, -1), d3 = (-1, 0), d4 = (0, 1), d5 = (1, 1)
    
    # Wait, let me verify. Starting at (r, 0):
    # Move in direction (0, -1) for r steps: (r, 0), (r, -1), ..., (r, -r)
    # Move in direction (-1, -1) for r steps: (r-1, -r-1), ..., (0, -2r)
    # Hmm, that doesn't seem right either.
    
    # Let me use a cleaner approach. The 6 corners of the hexagon of radius r are:
    # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
    
    # Wait, let me verify with r=1:
    # Corners: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # These should be the 6 points at hex distance 1.
    # Hex distance of (1,0): max(|1|,|0|,|1+0|) = max(1,0,1) = 1 ✓
    # Hex distance of (0,1): max(|0|,|1|,|0+1|) = max(0,1,1) = 1 ✓
    # Hex distance of (-1,1): max(|-1|,|1|,|-1+1|) = max(1,1,0) = 1 ✓
    # Hex distance of (-1,0): max(|-1|,|0|,|-1+0|) = max(1,0,1) = 1 ✓
    # Hex distance of (0,-1): max(|0|,|-1|,|0-1|) = max(0,1,1) = 1 ✓
    # Hex distance of (1,-1): max(|1|,|-1|,|1-1|) = max(1,1,0) = 1 ✓
    
    # Great! So the hex corners for radius r are:
    # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
    
    # And the edges connect consecutive corners.
    
    # For ring r, we traverse the 6 edges:
    # Edge 0: (r, 0) to (0, r): step (-1, 1), r steps
    # Edge 1: (0, r) to (-r, r): step (-1, 0), r steps  
    # Edge 2: (-r, r) to (-r, 0): step (0, -1), r steps
    # Edge 3: (-r, 0) to (0, -r): step (1, -1), r steps
    # Edge 4: (0, -r) to (r, -r): step (1, 0), r steps
    # Edge 5: (r, -r) to (r, 0): step (0, 1), r steps
    
    # Each edge has r+1 points including endpoints, but we skip the starting point
    # of each edge (since it's the endpoint of the previous edge).
    # So each edge contributes r points, total 6r points per ring.
    
    def hex_spiral_points():
        """Generate points in hexagonal spiral order."""
        # Ring 0
        yield (0, 0)
        
        r = 1
        while True:
            # Start at (r, 0)
            a, b = r, 0
            for edge in range(6):
                # Step direction for each edge
                if edge == 0:
                    da, db = -1, 1
                elif edge == 1:
                    da, db = -1, 0
                elif edge == 2:
                    da, db = 0, -1
                elif edge == 3:
                    da, db = 1, -1
                elif edge == 4:
                    da, db = 1, 0
                else:  # edge == 5
                    da, db = 0, 1
                
                for step in range(r):
                    a += da
                    b += db
                    yield (a, b)
            r += 1
    
    # Generate spiral points and take first 32
    gen = hex_spiral_points()
    for _ in range(32):
        a, b = next(gen)
        points.append([a, b])
    
    return points
# EVOLVE-BLOCK-END
