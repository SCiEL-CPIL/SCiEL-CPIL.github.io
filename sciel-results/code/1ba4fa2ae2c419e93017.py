import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # Pre-computed optimal hexagonal spiral: rings 0, 1, 2 (full) + 13 from ring 3
    # This achieves the Harborth optimum of 76 unit-distance pairs for n=32
    # Points are ordered by hex distance, then by angle for determinism
    
    # Ring 0 (1 point)
    pts = [(0, 0)]
    
    # Ring 1 (6 points): hex distance 1
    # Neighbors of origin: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    pts.extend(ring1)
    
    # Ring 2 (12 points): hex distance 2
    # Generate all points with hex_dist == 2
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    ring2 = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) == 2:
                ring2.append((a, b))
    pts.extend(ring2)
    
    # Ring 3 (need 13 of 18 points): hex distance 3
    # Sort by angle to get a consistent "spiral" subset
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # Sort ring3 by angle (atan2 of Cartesian coords)
    def angle_key(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    ring3.sort(key=angle_key)
    pts.extend(ring3[:13])
    
    # Now we have exactly 32 points
    assert len(pts) == 32, f"Expected 32 points, got {len(pts)}"
    
    # Verify all points are distinct
    assert len(set(pts)) == 32, "Points are not distinct"
    
    # Verify bounds
    for a, b in pts:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    # Count edges to verify we achieve 76
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    current_edges = count_edges(pts)
    
    # If we already have 76 edges (the optimum), return immediately
    if current_edges >= 76:
        return [list(p) for p in pts]
    
    # Local search polish: try to improve edge count
    # Convert to list of lists for mutability
    result = [list(p) for p in pts]
    point_set = set((p[0], p[1]) for p in result)
    
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(result)):
            if time.time() > deadline:
                break
            ax, ay = result[i][0], result[i][1]
            # Try moving point i to nearby empty positions
            for da in range(-2, 3):
                for db in range(-2, 3):
                    if da == 0 and db == 0:
                        continue
                    nx, ny = ax + da, ay + db
                    if (nx, ny) in point_set:
                        continue
                    if abs(nx) > 64 or abs(ny) > 64:
                        continue
                    
                    # Calculate delta in edge count
                    old_point = (ax, ay)
                    new_point = (nx, ny)
                    
                    edges_lost = 0
                    edges_gained = 0
                    for j in range(len(result)):
                        if j == i:
                            continue
                        if dist_sq(old_point, tuple(result[j])) == 1:
                            edges_lost += 1
                        if dist_sq(new_point, tuple(result[j])) == 1:
                            edges_gained += 1
                    
                    delta = edges_gained - edges_lost
                    if delta > 0:
                        result[i] = [nx, ny]
                        point_set.remove(old_point)
                        point_set.add(new_point)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return result
