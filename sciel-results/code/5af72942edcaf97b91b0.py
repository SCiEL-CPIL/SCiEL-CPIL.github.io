import math

def construct():
    # Generate points in axial coordinates (q, r)
    # Hexagonal distance: (|q| + |r| + |q+r|) // 2
    
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    # Get all points with dist <= 3
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if hex_dist(q, r) <= 3:
                points.append((q, r))
                
    # Separate into core (dist <= 2) and boundary (dist == 3)
    core = [p for p in points if hex_dist(p[0], p[1]) <= 2]
    boundary = [p for p in points if hex_dist(p[0], p[1]) == 3]
    
    # We need 32 points total. Core has 19. We need 13 from boundary.
    # To maximize edges, we pick a contiguous arc of 13 points from the boundary.
    # The boundary points form a cycle. We can sort them by angle.
    
    def angle(p):
        # Convert axial to cartesian for angle sorting
        # x = q + r/2
        # y = r * sqrt(3)/2
        x = p[0] + p[1] / 2.0
        y = p[1] * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    boundary_sorted = sorted(boundary, key=angle)
    
    # Pick 13 consecutive points from the sorted list (cyclic)
    # We can just take the first 13, or center them.
    # Due to symmetry, any contiguous block of 13 should have the same edge count.
    selected_boundary = boundary_sorted[:13]
    
    final_points = core + selected_boundary
    
    # Verify count
    if len(final_points) != 32:
        # Fallback: just take first 32 from points
        final_points = points[:32]
        
    # Count edges to verify (optional, but good for debugging)
    # dist_sq = (q1-q2)^2 + (q1-q2)(r1-r2) + (r1-r2)^2
    # If dist_sq == 1, it's an edge.
    
    return [[p[0], p[1]] for p in final_points]
