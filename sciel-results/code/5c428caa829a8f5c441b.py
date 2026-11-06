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
    
    # Theoretical max is 76 (Harborth's bound for n=32)
    if edge_count == 76:
        return points
    
    # If not optimal, do a fast local search with a tight budget
    point_set = set(map(tuple, points))
    all_candidates = set(candidates)
    outside = list(all_candidates - point_set)
    
    # Quick local search: try each swap once, accept first improvement
    for i, p_in in enumerate(points):
        if time.time() > deadline:
            break
        p_in_t = tuple(p_in)
        for p_out in outside:
            if time.time() > deadline:
                break
            new_set = (point_set - {p_in_t}) | {tuple(p_out)}
            new_pts = [list(p) for p in new_set]
            new_ec = count_edges(new_pts)
            if new_ec > edge_count:
                points[i] = list(p_out)
                point_set = new_set
                edge_count = new_ec
                outside.remove(p_out)
                outside.append(p_in_t)
                if edge_count == 76:
                    return points
    
    return points
