import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin before 150s kill

    # Helper to count unit distance pairs
    def count_edges(points):
        p_set = set(points)
        edges = 0
        # Neighbors on triangular lattice: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
        # We only count each edge once by checking 3 specific neighbors: (1,0), (0,1), (1,-1)
        for a, b in p_set:
            if (a+1, b) in p_set: edges += 1
            if (a, b+1) in p_set: edges += 1
            if (a+1, b-1) in p_set: edges += 1
        return edges

    # Generate points in hexagonal spiral order
    def hex_spiral(n):
        points = [(0, 0)]
        if n == 1:
            return points
        
        # Directions for the 6 sides of the hexagon rings
        # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Actually, let's just iterate radius and angle or use a standard spiral algorithm.
        
        # Standard hex spiral:
        # Center (0,0)
        # For radius r = 1, 2, ...
        #   Start at (r, 0) ? No, let's use axial coordinates.
        #   Points on ring r are those with max(|a|, |b|, |a+b|) = r.
        #   We can generate them in order.
        
        # Easier: Just generate all points in a large hexagon and sort by distance from center, 
        # breaking ties by angle.
        
        max_r = 10 # Enough for 32 points
        all_points = []
        for a in range(-max_r, max_r + 1):
            for b in range(-max_r, max_r + 1):
                # Check if in hexagon of radius max_r
                if max(abs(a), abs(b), abs(a+b)) <= max_r:
                    # Calculate squared distance from center
                    d2 = a*a + a*b + b*b
                    # Calculate angle for tie-breaking
                    # Convert to cartesian: x = a + b/2, y = b*sqrt(3)/2
                    x = a + b/2.0
                    y = b * 0.8660254037844386
                    angle = math.atan2(y, x)
                    all_points.append((d2, angle, a, b))
        
        # Sort by distance, then angle
        all_points.sort(key=lambda x: (x[0], x[1]))
        
        # Take first n points
        selected = [(a, b) for d2, ang, a, b in all_points[:n]]
        return selected

    # Initial guess: Hexagonal spiral
    points = hex_spiral(32)
    current_edges = count_edges(points)
    
    # Target is 76
    if current_edges < 76:
        # Local search: try swapping a point with a nearby non-selected point
        p_set = set(points)
        # Generate candidate swaps
        # We can try to move a point from the "outside" of the current set to a "hole" or closer position
        # Or simply try all pairs (p_in, p_out) where p_out is close to p_in
        
        # Since n=32 is small, we can do a more exhaustive local search if needed.
        # But likely the spiral is already optimal.
        
        # Let's check if we can improve by simple moves.
        # If not, we return the spiral.
        
        # A simple improvement heuristic:
        # Try replacing each point with a neighbor that is not in the set, 
        # if it increases the edge count.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_points = list(points)
            p_set = set(new_points)
            for i in range(len(new_points)):
                a, b = new_points[i]
                # Try moving this point to a neighbor
                for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), 
                               (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2),
                               (1,1), (-1,-1), (2,1), (-2,-1), (1,2), (-1,-2)]:
                    na, nb = a+da, b+db
                    if (na, nb) in p_set:
                        continue
                    # Swap
                    new_points[i] = (na, nb)
                    p_set.remove((a, b))
                    p_set.add((na, nb))
                    new_edges = count_edges(p_set)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        points = new_points
                        p_set = set(points)
                        improved = True
                        break
                    else:
                        # Revert
                        new_points[i] = (a, b)
                        p_set.remove((na, nb))
                        p_set.add((a, b))
                if improved:
                    break

    # Convert to list of lists as required
    return [[a, b] for a, b in points]
