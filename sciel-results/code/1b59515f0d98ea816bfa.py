import math
import time

def construct():
    start_time = time.time()
    timeout = 140  # 120s limit, keep safety margin

    # Lattice neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        edges = 0
        n = len(points)
        # Use a set for faster lookup? No, n=32, O(n^2) is fine.
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Generate Hexagon of Radius 3
    # Points (a,b) such that max(|a|, |b|, |a+b|) <= 3
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex_points.append((a,b))
                
    # We have 37 points. Need 32.
    # Strategy: Remove 5 boundary points with lowest degree.
    # Boundary points are those with max(|a|, |b|, |a+b|) == 3.
    
    current_points = set(hex_points)
    
    # Function to get degree of a point in the current set
    def get_degree(p, point_set):
        deg = 0
        a, b = p
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                deg += 1
        return deg

    # Identify candidates for removal: Boundary points
    # We will iteratively remove the point with the lowest degree.
    # Since we only remove 5 points, and they are likely corners (deg 3) or sides (deg 4),
    # we can just sort by degree and remove.
    
    # However, removing one point changes the degree of its neighbors.
    # But since we are removing from the "outside in", and corners are not adjacent,
    # the degrees of the other corners remain 3.
    # So we can just pick any 5 corners.
    
    # Let's find the corners.
    corners = []
    for p in current_points:
        if get_degree(p, current_points) == 3:
            # Check if it's a "corner" in the sense of the hexagon geometry
            # Corners of R=3 hexagon: (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
            # Actually, any point with degree 3 in the full hexagon is a corner of the hexagon.
            # Let's verify.
            # (3,0) has neighbors (2,0), (2,1), (3,-1). All inside.
            # (2,1) is on the edge?
            # Max(2, 1, 3) = 3. Yes.
            # Does (2,1) have degree 3?
            # Neighbors of (2,1):
            # (3,1) - out (max 4)
            # (2,2) - out (max 4)
            # (1,2) - in (max 2)
            # (1,1) - in (max 2)
            # (2,0) - in (max 2)
            # (3,0) - in (max 3)
            # So (2,1) has 4 neighbors: (1,2), (1,1), (2,0), (3,0).
            # So (2,1) is a side point with degree 4.
            # Only the 6 extreme corners have degree 3.
            corners.append(p)
            
    # We have 6 corners. We need to remove 5.
    # Remove the first 5.
    to_remove = corners[:5]
    
    for p in to_remove:
        current_points.remove(p)
        
    points_list = list(current_points)
    
    # Verify count
    if len(points_list) != 32:
        # Fallback if logic failed
        points_list = hex_points[:32]
        
    # Calculate edges
    edges = count_edges(points_list)
    
    # If edges < 76, run local search to improve
    if edges < 76:
        best_points = points_list[:]
        best_edges = edges
        
        # Local Search: Single point move
        # Try moving each point to a neighbor position if it increases edges
        # Or swap with a point outside?
        # Moving is safer for maintaining distinctness.
        
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            current_edges = count_edges(best_points)
            for i in range(32):
                a, b = best_points[i]
                # Try moving to all 6 neighbors
                for da, db in neighbors:
                    new_a, new_b = a + da, b + db
                    new_p = (new_a, new_b)
                    
                    # Check bounds
                    if abs(new_a) > 64 or abs(new_b) > 64:
                        continue
                    
                    # Check collision
                    valid = True
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(new_p, best_points[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate change in edges
                    # Old edges involving i
                    old_edges_i = 0
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(best_points[i], best_points[j]) == 1:
                            old_edges_i += 1
                            
                    # New edges involving i
                    new_edges_i = 0
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(new_p, best_points[j]) == 1:
                            new_edges_i += 1
                            
                    delta = new_edges_i - old_edges_i
                    if delta > 0:
                        best_points[i] = new_p
                        current_edges += delta
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                        break # Restart loop with new configuration
                if improved:
                    break
                    
    return [[a, b] for a, b in best_points]
