"""Rewritten program: Harborth spiral for 32 pennies on triangular lattice."""
# EVOLVE-BLOCK-START
def construct():
    """
    Generate 32 lattice points (a, b) on the triangular lattice such that:
      - All points are distinct integer pairs with |a|, |b| <= 64
      - Mutual distance >= 1 (in Euclidean sense on the triangular lattice)
      - Number of pairs at distance exactly 1 is maximized (target: 76)
    
    The triangular lattice point a + b*omega where omega = exp(i*pi/3) = (1/2, sqrt(3)/2).
    Distance squared between (a1,b1) and (a2,b2) is:
      dx = a1 - a2, dy = b1 - b2
      dist^2 = dx^2 + dy^2 + dx*dy
    
    Adjacent pairs (distance 1) satisfy dx^2 + dy^2 + dx*dy = 1, which gives:
      (dx,dy) in {(1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)}
    
    We use a spiral growth pattern on the triangular lattice to achieve the
    Harborth optimum of 76 edges for n=32.
    
    Strategy: Build a compact hexagonal-ish cluster by growing a spiral.
    For the triangular lattice, a good compact shape is a "triangular number"
    arrangement or a hexagon. For n=32, we use a spiral that fills layers
    of a hexagonal region.
    
    The hexagonal numbers are 1, 7, 19, 37, ... so 32 falls between 19 and 37.
    We start with the center and add layers, stopping when we have 32 points.
    
    A better approach: use the known optimal configuration. For n=32 on the
    triangular lattice, the optimum is 76 edges. A spiral/hexagonal packing
    achieves this.
    
    Let's generate points by expanding rings around the origin:
    Ring 0: (0,0) -> 1 point
    Ring 1: 6 points at distance 1 from origin
    Ring 2: 12 points at distance 2 from origin (but only those not already counted)
    Ring 3: 18 points...
    
    Cumulative: 1, 7, 19, 37...
    For 32 points: take ring 0 (1) + ring 1 (6) + ring 2 (12) = 19, then 13 more from ring 3.
    
    But we need to be careful about which points to pick from ring 3 to maximize
    internal edges. Let's use a systematic spiral.
    
    Alternative: Use a known optimal pattern. The "Harborth spiral" on the
    triangular lattice for n=32 can be constructed by a specific ordering.
    
    Let me use a different approach: generate all points in a bounding box,
    then select 32 that form a connected subgraph with maximum edges.
    
    Actually, the simplest correct approach: use the hexagonal spiral.
    """
    import math
    
    # Directions on triangular lattice for the 6 neighbors
    # (da, db) such that da^2 + db^2 + da*db = 1
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate points using a spiral around the origin
    # We'll use the standard hexagonal spiral on the triangular lattice
    # Ring r has 6*r points (for r >= 1), ring 0 has 1 point
    
    points = [(0, 0)]
    
    if len(points) >= 32:
        return [[a, b] for a, b in points[:32]]
    
    # For each ring r >= 1, the 6*r points are at "distance" r in the
    # hexagonal metric. We need to enumerate them properly.
    # 
    # On the triangular lattice with coordinates (a,b), the hexagonal distance
    # from origin is max(|a|, |b|, |a+b|) ... let me verify.
    # Actually, the distance in the triangular lattice (minimum number of steps)
    # from (0,0) to (a,b) is: 
    #   d = max(|a|, |b|, |a+b|) if we use the right metric? 
    # Let me think again. The 6 neighbors are at (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1).
    # The distance from (0,0) to (a,b) is the minimum number of steps.
    # This is equivalent to: let c = -(a+b), then distance = (|a| + |b| + |c|) / 2? No.
    # 
    # Actually for the triangular lattice with these neighbors, the distance is:
    # d(a,b) = max(|a|, |b|, |a+b|)
    # Let's verify: d(1,0) = max(1,0,1) = 1. Correct.
    # d(0,1) = max(0,1,1) = 1. Correct.
    # d(1,-1) = max(1,1,0) = 1. Correct.
    # d(2,0) = max(2,0,2) = 2. Correct (two steps along (1,0)).
    # d(1,1) = max(1,1,2) = 2. Path: (0,0)->(0,1)->(1,1)? No, (1,1) is not a neighbor of (0,1).
    # (0,1) neighbors: (1,1),(0,2),(-1,2),(-1,1),(0,0),(1,0). So (1,1) is neighbor of (0,1).
    # So d(1,1) = 2. And max(1,1,2)=2. Correct.
    
    # So ring r consists of all (a,b) with max(|a|,|b|,|a+b|) = r.
    # Number of points in ring r: 6r for r >= 1.
    
    def hex_ring(r):
        """Generate all 6*r points in hexagonal ring r."""
        if r == 0:
            return [(0, 0)]
        
        # The ring r is a hexagon. We can generate it by walking around.
        # Start at (r, 0) and walk along the 6 sides.
        # Side 1: from (r, 0) to (0, r): step (-1, 1) r times? 
        # Let me think of the 6 vertices of the hexagon:
        # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
        # 
        # Side 1: (r,0) -> (0,r): direction (-1,1), r steps
        # Side 2: (0,r) -> (-r,r): direction (-1,0), r steps
        # Side 3: (-r,r) -> (-r,0): direction (0,-1), r steps
        # Side 4: (-r,0) -> (0,-r): direction (1,-1), r steps
        # Side 5: (0,-r) -> (r,-r): direction (1,0), r steps
        # Side 6: (r,-r) -> (r,0): direction (0,1), r steps
        
        sides = [
            (-1, 1),  # (r,0) to (0,r)
            (-1, 0),  # (0,r) to (-r,r)
            (0, -1),  # (-r,r) to (-r,0)
            (1, -1),  # (-r,0) to (0,-r)
            (1, 0),   # (0,-r) to (r,-r)
            (0, 1),   # (r,-r) to (r,0)
        ]
        
        pts = []
        a, b = r, 0
        for da, db in sides:
            for _ in range(r):
                a += da
                b += db
                pts.append((a, b))
        return pts
    
    # Build the spiral: add rings until we have >= 32 points
    ring = 1
    while len(points) < 32:
        ring_pts = hex_ring(ring)
        # Add points from this ring until we have 32
        for p in ring_pts:
            if len(points) >= 32:
                break
            points.append(p)
        ring += 1
    
    # We have exactly 32 points now
    points = points[:32]
    
    # Verify all coordinates are within bounds
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
