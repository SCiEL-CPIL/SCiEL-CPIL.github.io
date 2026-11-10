"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice to maximize unit-distance edges.
    Uses a hexagonal spiral filling to create a compact cluster.
    """
    points = []
    
    # Helper to generate points in a hexagonal ring of radius r
    # Hex distance in axial coordinates (a, b) is max(|a|, |b|, |a+b|)
    # But standard hex grid coordinates often use cube (x, y, z) with x+y+z=0.
    # Here we use axial a, b.
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    
    # We want to generate points in order of increasing distance from origin.
    # Distance squared: a^2 + ab + b^2.
    # However, "hex distance" (graph distance) is better for rings.
    # Hex distance d(a,b) = max(|a|, |b|, |a+b|) is NOT correct for this basis?
    # Let's check.
    # Basis: u=(1,0), v=(0,1).
    # Point (a,b) = a*u + b*v.
    # Distance to origin in lattice steps:
    # It is known that for this basis, the hex distance is max(|a|, |b|, |a+b|) if the basis is 60 deg?
    # Wait.
    # If u and v are 60 deg apart.
    # (1,0) dist 1. max(1,0,1)=1.
    # (0,1) dist 1. max(0,1,1)=1.
    # (1,1) = u+v. dist?
    # |u+v|^2 = 1+1+1 = 3. Euclidean dist sqrt(3).
    # Graph dist from (0,0) to (1,1):
    # (0,0) -> (1,0) -> (1,1)?
    # (1,0) to (1,1) is (0,1) step. Yes.
    # So graph dist is 2.
    # max(1,1,2) = 2.
    # So yes, hex distance is max(|a|, |b|, |a+b|).
    
    # We will collect points by ring.
    # Ring 0: (0,0)
    # Ring 1: dist 1
    # Ring 2: dist 2
    # Ring 3: dist 3
    
    # To maintain a connected "spiral" or compact shape, we should order the points.
    # Simply taking all points with dist <= 2 gives 19 points.
    # We need 13 more from dist 3.
    # To maximize edges, we should pick points that are adjacent to the existing set.
    # All points in ring 3 are adjacent to ring 2?
    # A point (a,b) in ring 3 has max(|a|,|b|,|a+b|)=3.
    # It will have neighbors in ring 2 and ring 3.
    # If we pick a contiguous arc of ring 3, they will be connected to each other and to ring 2.
    
    # Let's generate all points with hex_dist <= 3.
    # Total 37 points.
    # Sort by (hex_dist, angle) to create a spiral.
    
    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a+b))
    
    def angle_key(a, b):
        # Convert to cartesian for angle
        # x = a + b/2
        # y = b * sqrt(3)/2
        import math
        x = a + 0.5 * b
        y = math.sqrt(3) / 2 * b
        return math.atan2(y, x)
        
    candidates = []
    # Search range large enough to cover ring 3
    # Max coord is 3.
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = hex_dist(a, b)
            if d <= 3:
                candidates.append((d, angle_key(a, b), a, b))
                
    # Sort by distance, then angle
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Take first 32
    selected = []
    for d, ang, a, b in candidates:
        if len(selected) == 32:
            break
        selected.append([a, b])
        
    return selected

# EVOLVE-BLOCK-END
