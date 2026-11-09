import random
import time

def construct():
    # Start with a hexagonal spiral
    points = []
    # Generate spiral
    # Center
    points.append((0,0))
    # Rings
    # Direction vectors
    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # We need 32 points.
    # Ring 0: 1
    # Ring 1: 6 (total 7)
    # Ring 2: 12 (total 19)
    # Ring 3: 18 (total 37)
    # We take all from Ring 0, 1, 2 (19 points)
    # And 13 from Ring 3.
    
    spiral = [(0,0)]
    # Generate ring 1
    # Start at (1,0), go to (0,1), etc?
    # Standard spiral:
    # (1,0), (1,1), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) ... wait, this is not a simple ring.
    
    # Let's just generate all points with max(|a|,|b|,|a+b|) <= 3
    # Sort them by distance from center, then angle?
    # Or just take the first 32 in a spiral order.
    
    # Simpler: Generate all points in R=3 hexagon.
    # Sort by "spiral index".
    # Spiral index can be approximated by distance, then angle.
    
    candidates = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                candidates.append((a,b))
                
    # Sort by distance from origin
    def dist_sq(p):
        a, b = p
        return a*a + b*b + a*b
        
    candidates.sort(key=dist_sq)
    
    # The first 32 points in this sorted list will be the compact hexagon.
    # Note: Points with same distance are ordered arbitrarily by sort.
    # This might not be the optimal 32 points.
    # But we will use local search to fix it.
    
    current_set = set(candidates[:32])
    
    # Calculate initial edges
    def count_edges(s):
        edges = 0
        pts = list(s)
        for i in range(len(pts)):
            a1, b1 = pts[i]
            for j in range(i+1, len(pts)):
                a2, b2 = pts[j]
                da = a1 - a2
                db = b1 - b2
                # Check if distance is 1
                # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
                if (da, db) in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    edges += 1
        return edges
        
    current_edges = count_edges(current_set)
    
    # Local Search
    # Time limit guard
    start_time = time.time()
    timeout = 10.0 # 10 seconds is plenty for 32 points
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        pts = list(current_set)
        random.shuffle(pts)
        for p in pts:
            # Try moving p to a neighbor
            a, b = p
            neighbors = [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]
            for np in neighbors:
                if np in current_set:
                    continue
                # Calculate gain
                # Remove p
                # Add np
                # Gain = (neighbors of np in current_set - {p}) - (neighbors of p in current_set - {np})
                # Note: np is not in current_set, so neighbors of np in current_set are just neighbors in current_set.
                # Neighbors of p in current_set are neighbors in current_set.
                
                # Count neighbors of p in current_set
                old_deg = 0
                for na, nb in neighbors:
                    if (na, nb) in current_set:
                        old_deg += 1
                        
                # Count neighbors of np in current_set
                # np neighbors are the same relative offsets
                new_deg = 0
                for na, nb in neighbors:
                    # Note: the neighbors of np are different coordinates
                    # If p is (a,b), np is (a+1, b).
                    # Neighbors of np are (a+2, b), (a, b), (a+1, b+1), (a+1, b-1), (a+2, b-1), (a, b+1).
                    # We need to check if these are in current_set.
                    pass # This is getting complicated to hardcode.
                    
                # Easier: just simulate the move and count edges?
                # 32 points, 32*6 moves. 192 moves.
                # Counting edges is O(N^2) = 1024.
                # Total 192 * 1024 ~ 200k ops. Very fast.
                
                new_set = current_set.copy()
                new_set.remove(p)
                new_set.add(np)
                
                new_edges = count_edges(new_set)
                
                if new_edges > current_edges:
                    current_set = new_set
                    current_edges = new_edges
                    improved = True
                    break # Restart loop with new set
                    
    # Convert to list of lists
    return [list(p) for p in current_set]
