import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    # We need exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    # Points are a + b*omega, omega = exp(i*pi/3)
    # Mutual distance at least 1 (in lattice units, i.e., adjacent points have distance 1)
    # Maximize number of pairs at distance exactly 1.
    # Harborth's theorem says optimum is floor(3n - sqrt(12n - 3)) = 76 for n=32.

    # Distance squared on triangular lattice:
    # For points (a1, b1) and (a2, b2), let da = a1-a2, db = b1-b2
    # |z|^2 = da^2 + db^2 + da*db  (since |omega|^2 = 1 and Re(omega) = 1/2)
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin for point (a, b) in this coordinate system.
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Strategy: Generate a compact set of 32 points.
    # A regular hexagon of radius 2 has 1 + 6 + 12 = 19 points.
    # We need 32 points, so we need 13 more from the next ring (radius 3).
    # The ring at radius 3 has 6*3 = 18 points.
    # We should pick 13 of them such that they maximize internal edges and edges to the inner part.
    
    # Let's first generate the core hexagon of radius 2 (19 points).
    core = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                core.add((a, b))
    
    # Now generate the ring at radius 3.
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # We need to choose 13 points from ring3 to add to the 19 core points.
    # Total points = 32.
    # We want to maximize total edges.
    # Edges within core are fixed.
    # Edges within the chosen subset of ring3 depend on which 13 we pick.
    # Edges between core and chosen ring3 points depend on which 13 we pick.
    
    # Let's compute the number of edges within the core first.
    core_list = list(core)
    edges_core = count_edges(core_list)
    
    # Let's order the ring3 points by their angle to form the cycle.
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    ring3.sort(key=angle)
    
    # Now, for each starting index i, take ring3[i], ring3[(i+1)%18], ..., ring3[(i+12)%18]
    best_score = -1
    best_block = None
    
    for i in range(18):
        if time.time() > deadline:
            break
        subset = [ring3[(i + k) % 18] for k in range(13)]
        full_set = list(core) + subset
        
        # Verify all points are distinct
        if len(set(map(tuple, full_set))) != 32:
            continue
        
        # Verify bounds
        if any(abs(p[0]) > 64 or abs(p[1]) > 64 for p in full_set):
            continue
        
        edges = count_edges(full_set)
        
        if edges > best_score:
            best_score = edges
            best_block = [list(p) for p in full_set]
    
    if best_score == 76:
        return best_block

    # Fallback: Generate the first 32 points of the hexagonal spiral
    dirs = [
        (1, 0),   # East
        (0, 1),   # North-East
        (-1, 1),  # North-West
        (-1, 0),  # West
        (0, -1),  # South-West
        (1, -1)   # South-East
    ]
    
    points = [(0, 0)]
    
    step_size = 1
    dir_index = 0
    
    while len(points) < 32:
        if time.time() > deadline:
            break
        steps_needed = dir_index // 2 + 1
        da, db = dirs[dir_index % 6]
        
        for _ in range(steps_needed):
            if len(points) >= 32:
                break
            last_a, last_b = points[-1]
            new_a = last_a + da
            new_b = last_b + db
            points.append((new_a, new_b))
        
        dir_index += 1
        
    return [[a, b] for a, b in points[:32]]
