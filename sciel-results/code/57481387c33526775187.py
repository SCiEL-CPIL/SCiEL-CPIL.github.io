"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    
    The triangular lattice has basis vectors e1 = (1,0) and e2 = (1/2, sqrt(3)/2).
    A point is represented as (a, b) meaning a*e1 + b*e2.
    
    The six unit neighbors of (a,b) are:
      (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    For n=32, a hexagonal spiral attains the Harborth bound of 76 unit distances.
    """
    # The six neighbor offsets
    NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Generate hexagonal spiral: center + rings 1, 2, 3, ...
    # Ring r has 6*r points. Cumulative: 1, 7, 19, 37, ...
    # For n=32: center (1) + ring 1 (6) + ring 2 (12) + ring 3 (first 13 of 18) = 32
    
    def hex_spiral(n_points):
        """Generate the first n_points of the hexagonal spiral in proper spiral order."""
        points = [(0, 0)]
        if n_points <= 1:
            return points
        
        # For each ring r, we traverse 6 sides of length r
        # Side directions and starting corners:
        # The hex ring at distance r consists of 6 segments.
        # We start at (r, 0) and go around.
        # 
        # Proper spiral ordering for ring r:
        # Start at (r, 0)
        # Side 0: move in direction (0, 1) for r steps: (r,0)->(r,1)->...->(r,r) ... wait
        # Let me think about this more carefully.
        #
        # The 6 corners of ring r are:
        # C0 = (r, 0)
        # C1 = (0, r)
        # C2 = (-r, r)
        # C3 = (-r, 0)
        # C4 = (0, -r)
        # C5 = (r, -r)
        #
        # Side from C0 to C1: direction (-1, 1), r steps
        # Side from C1 to C2: direction (-1, 0), r steps
        # Side from C2 to C3: direction (0, -1), r steps
        # Side from C3 to C4: direction (1, -1), r steps
        # Side from C4 to C5: direction (1, 0), r steps
        # Side from C5 to C0: direction (0, 1), r steps
        #
        # For the spiral, we include C0 but not the endpoint of each side (to avoid duplication)
        # So each side contributes r points (including start, excluding end)
        
        ring = 1
        while len(points) < n_points:
            # Corners
            corners = [
                (ring, 0),
                (0, ring),
                (-ring, ring),
                (-ring, 0),
                (0, -ring),
                (ring, -ring)
            ]
            # Side directions (from corner i to corner i+1)
            side_dirs = [
                (-1, 1),   # C0 -> C1
                (-1, 0),   # C1 -> C2
                (0, -1),   # C2 -> C3
                (1, -1),   # C3 -> C4
                (1, 0),    # C4 -> C5
                (0, 1)     # C5 -> C0
            ]
            
            for side in range(6):
                if len(points) >= n_points:
                    break
                cx, cy = corners[side]
                dx, dy = side_dirs[side]
                # Walk r steps from corner, including start, excluding end
                for step in range(ring):
                    if len(points) >= n_points:
                        break
                    px = cx + dx * step
                    py = cy + dy * step
                    points.append((px, py))
            
            ring += 1
        
        return points[:n_points]
    
    # Generate the 32-point hexagonal spiral
    points = hex_spiral(32)
    
    # Verify: count unit-distance pairs using neighbor lookup
    point_set = set(points)
    edge_count = 0
    for a, b in points:
        for da, db in NEIGHBORS:
            if (a + da, b + db) in point_set:
                edge_count += 1
    edge_count //= 2  # Each edge counted twice
    
    # The hex spiral for n=32 should give exactly 76 edges (Harborth bound)
    # Return the result
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
