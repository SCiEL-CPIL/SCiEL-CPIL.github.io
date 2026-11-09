import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Generate initial hexagonal spiral points
    # Axial coordinates (a, b) for triangular lattice
    # Distance squared: (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    # We want distance squared == 1
    
    def dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db

    def get_spiral_points(n):
        points = [(0, 0)]
        if n == 1:
            return points
        
        # Directions for the 6 sides of the hexagon
        # In axial coords, neighbors of (0,0) are:
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Let's define the 6 directions
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        # We will generate rings k=1, 2, ...
        # Ring k has 6k points.
        # Start point of ring k is k steps in direction (1, -1) from origin?
        # Let's use a standard spiral generation.
        # (0,0)
        # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # This is counter-clockwise?
        
        # A simpler way: iterate through all lattice points in increasing "hex distance"
        # Hex distance from origin: max(abs(a), abs(b), abs(-a-b)) ? No.
        # In axial coords, distance to origin is (abs(a) + abs(b) + abs(a+b)) / 2
        
        candidates = []
        max_coord = 10
        for a in range(-max_coord, max_coord + 1):
            for b in range(-max_coord, max_coord + 1):
                # Check if within bounds |a|,|b| <= 64
                if abs(a) > 64 or abs(b) > 64:
                    continue
                d = (abs(a) + abs(b) + abs(a + b)) // 2
                candidates.append((d, a, b))
        
        # Sort by distance, then by angle to ensure spiral-like ordering
        # Angle: atan2(sqrt(3)/2 * b, a + b/2) ?
        # In complex plane: z = a + b * exp(i*pi/3) = a + b*(1/2 + i*sqrt(3)/2)
        # x = a + b/2, y = b*sqrt(3)/2
        import math
        def sort_key(item):
            d, a, b = item
            x = a + b / 2.0
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x)
            return (d, angle)
            
        candidates.sort(key=sort_key)
        
        # Extract first n points
        # Note: (0,0) is included in candidates with d=0
        pts = [(a, b) for d, a, b in candidates[:n]]
        return pts

    points = get_spiral_points(32)
    
    def count_edges(pts):
        count = 0
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(pts[i], pts[j]) == 1:
                    count += 1
        return count

    current_edges = count_edges(points)
    
    # If we haven't reached 76, try to optimize
    # The points in the outermost ring (distance 3) are the most flexible.
    # Identify points at hex distance 3
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a + b)) // 2

    # We have 32 points.
    # Distances: 
    # d=0: 1 pt
    # d=1: 6 pts (total 7)
    # d=2: 12 pts (total 19)
    # d=3: 13 pts (total 32)
    
    # We can try to swap points at d=3 with other points at d=3 or d=4 to improve edges.
    # However, the spiral selection is usually optimal for this n.
    # Let's verify if 76 is reached.
    
    if current_edges < 76:
        # Simple local search: try to move one of the d=3 points to a neighbor position
        # that is not occupied, and check if edge count increases.
        # Or swap two d=3 points.
        
        occupied = set(points)
        
        # Identify the d=3 points
        d3_pts = [p for p in points if hex_dist(p) == 3]
        
        # Generate candidate moves for each d3 point
        # Neighbors of a point (a,b) are the 6 axial neighbors
        neighbor_dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        
        improved = True
        max_iterations = 5000
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline:
            improved = False
            iter_count += 1
            for i, p in enumerate(d3_pts):
                a, b = p
                for da, db in neighbor_dirs:
                    new_p = (a + da, b + db)
                    # Check bounds
                    if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                        continue
                    # Check if occupied
                    if new_p in occupied:
                        continue
                    
                    # Try moving p to new_p
                    new_points = list(points)
                    new_points[i] = new_p
                    new_occupied = set(new_points)
                    
                    new_edges = count_edges(new_points)
                    if new_edges > current_edges:
                        points = new_points
                        occupied = new_occupied
                        d3_pts[i] = new_p
                        current_edges = new_edges
                        improved = True
                        # Restart loop to find more improvements
                        break
                if improved:
                    break
            
            # Also try swapping two d3 points
            if not improved and len(d3_pts) >= 2:
                for i in range(len(d3_pts)):
                    for j in range(i+1, len(d3_pts)):
                        p_i = d3_pts[i]
                        p_j = d3_pts[j]
                        # Swap
                        new_points = list(points)
                        new_points[i] = p_j
                        new_points[j] = p_i
                        new_edges = count_edges(new_points)
                        if new_edges > current_edges:
                            points = new_points
                            d3_pts[i] = p_j
                            d3_pts[j] = p_i
                            current_edges = new_edges
                            improved = True
                            break
                    if improved:
                        break

    # Final check
    # Ensure distinctness and bounds
    final_points = []
    seen = set()
    for p in points:
        if p not in seen and abs(p[0]) <= 64 and abs(p[1]) <= 64:
            final_points.append(p)
            seen.add(p)
            
    # If we lost points due to some error, refill with spiral points
    if len(final_points) < 32:
        spiral_pts = get_spiral_points(32)
        for p in spiral_pts:
            if p not in seen:
                final_points.append(p)
                seen.add(p)
                if len(final_points) == 32:
                    break

    # Convert to list of lists
    return [[a, b] for a, b in final_points[:32]]
