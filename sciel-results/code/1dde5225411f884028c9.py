import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145  # 5 seconds margin
    
    # Define neighbors
    # (a, b) neighbors: (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Candidate pool: points within a certain radius
    # We need 32 points. A hexagon of radius 4 has 37 points.
    # So radius 4 is sufficient.
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            # Check if distance is small enough
            # a^2 + ab + b^2 <= 16 (radius 4)
            if a*a + a*b + b*b <= 16:
                candidates.append((a, b))
                
    selected = set()
    selected_list = []
    
    # Greedy selection
    for _ in range(32):
        if time.time() > deadline:
            break
            
        best_point = None
        best_score = -1
        best_dist = float('inf')
        
        for p in candidates:
            if p in selected:
                continue
                
            # Count edges to selected points
            edges = 0
            for n in neighbors:
                if (p[0]+n[0], p[1]+n[1]) in selected:
                    edges += 1
                    
            # Tie-breaker: closer to center is better
            dist = p[0]*p[0] + p[0]*p[1] + p[1]*p[1]
            
            if edges > best_score or (edges == best_score and dist < best_dist):
                best_score = edges
                best_dist = dist
                best_point = p
                
        if best_point is not None:
            selected.add(best_point)
            selected_list.append(best_point)
            
    # If we didn't select 32 points (should not happen), pad with arbitrary points
    if len(selected_list) < 32:
        for p in candidates:
            if p not in selected:
                selected.add(p)
                selected_list.append(p)
                if len(selected_list) == 32:
                    break
                    
    return [list(p) for p in selected_list]
