import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate all points in a box
    # Radius 3 hexagon fits in |a|,|b|,|a+b| <= 3, so a,b in [-3,3] suffices
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist([a, b]) <= 3:
                all_points.append([a, b])

    # Inner: radius <= 2 (19 points)
    inner = [p for p in all_points if hex_dist(p) <= 2]
    # Outer: radius == 3 (18 points)
    outer = [p for p in all_points if hex_dist(p) == 3]

    assert len(inner) == 19, f"inner={len(inner)}"
    assert len(outer) == 18, f"outer={len(outer)}"

    # We need 32 total, so pick 13 from outer
    # Start with inner (19 points), then add 13 outer points to maximize edges
    
    # Exhaustive search over all subsets of 13 from the 18 outer points
    # C(18, 13) = 8568, which is small enough to enumerate fully
    best_set = None
    best_edges = -1
    k_needed = 32 - len(inner)  # 13
    
    # Pre-compute edges within inner
    inner_edges = count_edges(inner)
    
    # Pre-compute adjacency of outer points to inner and other outer points
    # For speed, we can just count edges for each candidate
    
    from itertools import combinations
    
    for combo in combinations(outer, k_needed):
        candidate = inner + list(combo)
        # Optimization: only count edges involving outer points + inner-inner
        # Edges = inner_edges + edges(combo) + edges(combo, inner)
        
        # Count edges within combo
        e_combo = 0
        for i in range(len(combo)):
            for j in range(i+1, len(combo)):
                if dist_sq(combo[i], combo[j]) == 1:
                    e_combo += 1
        
        # Count edges between combo and inner
        e_cross = 0
        for p in combo:
            for q in inner:
                if dist_sq(p, q) == 1:
                    e_cross += 1
        
        total_edges = inner_edges + e_combo + e_cross
        
        if total_edges > best_edges:
            best_edges = total_edges
            best_set = candidate

    # Verify we have exactly 32 distinct points
    result = best_set
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64

    if time.time() > deadline:
        return result

    return result
