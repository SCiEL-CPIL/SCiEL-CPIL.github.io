import random
import math
import time

def construct():
    # Lattice neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        edges = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Deterministic Spiral Construction
    # Generate points in order of distance from center, then angle.
    # We need 32 points.
    
    points = []
    # Add center
    points.append((0,0))
    
    # Add rings
    # Ring r has 6r points.
    # We need 31 more points.
    # Ring 1: 6 pts. Total 7.
    # Ring 2: 12 pts. Total 19.
    # Ring 3: 18 pts. We need 13 more. Total 32.
    
    # Generate Ring 1
    for step in range(6):
        # Direction vectors for the 6 sides of the hexagon
        # Side 1: (1,0) to (1,-1)? 
        # Let's just iterate through the perimeter of the hexagon of radius r.
        # Corners: (r,0), (0,r), (-r,r), (-r,0), (0,-r), (r,-r)
        # We can generate the perimeter points.
        pass # We'll do this in a helper or inline
        
    # Simpler: Generate all points in hexagon R=3, sort by distance, then take first 32?
    # Distance from center in lattice metric: max(|a|, |b|, |a+b|)
    # If we sort by this distance, we get rings.
    # Within a ring, order matters for edge count of the subset.
    # If we take the first 13 of Ring 3 in spiral order, we get a contiguous arc.
    
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex_points.append((a,b))
                
    # Sort by distance from origin
    def lattice_dist(p):
        return max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))
        
    # Secondary sort: angle or spiral order
    # To get a contiguous arc, we can sort by angle?
    # Or just take the points in the order they appear in a spiral traversal.
    
    # Let's generate spiral order explicitly.
    # Start at (0,0).
    # Move to (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- this is Ring 1?
    # No, (1,0) is dist 1. (0,1) is dist 1.
    # The 6 neighbors of (0,0) are Ring 1.
    # Then Ring 2.
    
    # Algorithm for spiral:
    # 1. (0,0)
    # 2. (1,0)
    # 3. (1,-1)
    # 4. (0,-1)
    # 5. (-1,0)
    # 6. (-1,1)
    # 7. (0,1)
    # This is Ring 1 (6 pts).
    # Then Ring 2 starts from (2,0)?
    # Actually, standard hex spiral:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-2)? No.
    
    # Let's just use the sorted list and break ties by a deterministic angle.
    # Angle = atan2(b + a/2, a) ?
    # In triangular lattice, complex number is a + b*exp(i*pi/3).
    # x = a + b/2, y = b*sqrt(3)/2.
    # Angle = atan2(y, x).
    
    def get_angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3)/2.0
        return math.atan2(y, x)
        
    # Sort by distance, then angle
    hex_points.sort(key=lambda p: (lattice_dist(p), get_angle(p)))
    
    # Take first 32
    initial_points = hex_points[:32]
    
    # Local Search to ensure optimality
    best_points = initial_points[:]
    best_edges = count_edges(best_points)
    
    # Deterministic Local Search
    # Try all single moves.
    improved = True
    while improved:
        improved = False
        current_edges = count_edges(best_points)
        for i in range(32):
            a, b = best_points[i]
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
                old_edges_i = 0
                for j in range(32):
                    if j == i: continue
                    if dist_sq(best_points[i], best_points[j]) == 1:
                        old_edges_i += 1
                    
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
                    break
            if improved:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            
    return [[a, b] for a, b in best_points]
