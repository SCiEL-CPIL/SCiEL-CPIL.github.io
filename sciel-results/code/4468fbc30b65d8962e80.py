import math
import time

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    n_points = 32
    
    # Helper to compute hex distance
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2
        
    # Helper to compute angle for tie-breaking
    def angle(p):
        a, b = p
        x = a + 0.5 * b
        y = math.sqrt(3) / 2.0 * b
        return math.atan2(y, x)

    # Neighbors at distance 1
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def count_edges(pts):
        lookup = set(map(tuple, pts))
        ec = 0
        for p in pts:
            a, b = p
            for da, db in neighbors:
                if (a+da, b+db) in lookup:
                    ec += 1
        return ec // 2

    # Generate candidate points
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            candidates.append((a, b))
            
    # Sort by hex distance, then by angle
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    
    # Take the first 32
    points = [list(p) for p in candidates[:n_points]]
    
    edge_count = count_edges(points)
    
    # Theoretical max is 76
    if edge_count == 76:
        return points
    
    # Local search: try swapping a point in with a point out to improve edge count
    lookup = set(map(tuple, points))
    point_set = set(map(tuple, points))
    
    # Build list of points not in our set but within range
    all_candidates = set(candidates)
    outside = list(all_candidates - point_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        for i, p_in in enumerate(points):
            for p_out in outside:
                # Try swapping p_in with p_out
                new_set = (point_set - {tuple(p_in)}) | {tuple(p_out)}
                new_pts = [list(p) for p in new_set]
                new_ec = count_edges(new_pts)
                gain = new_ec - edge_count
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, p_in, p_out)
        
        if best_swap is not None:
            i, p_in, p_out = best_swap
            points[i] = list(p_out)
            point_set = set(map(tuple, points))
            edge_count += best_gain
            improved = True
            # Update outside list
            outside = [p for p in outside if tuple(p) != tuple(p_out)]
            outside.append(tuple(p_in))
    
    return points
