import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0

    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b * 0.5
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    candidates = []
    R = 4
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])
    
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates[:32]]
    
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def count_edges(points):
        lookup = set()
        for p in points:
            lookup.add((p[0], p[1]))
        count = 0
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a+da, b+db) in lookup:
                    count += 1
        return count // 2

    def verify(points):
        if len(points) != 32:
            return False
        seen = set()
        for p in points:
            a, b = p
            if abs(a) > 64 or abs(b) > 64:
                return False
            if (a, b) in seen:
                return False
            seen.add((a, b))
        for i in range(32):
            for j in range(i+1, 32):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True

    if not verify(result):
        # Fallback
        candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                if hex_dist([a, b]) <= 6:
                    candidates.append([a, b])
        candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
        result = [p[:] for p in candidates[:32]]
        if not verify(result):
            result = []
            for a in range(-4, 5):
                for b in range(-4, 5):
                    if len(result) < 32:
                        result.append([a, b])
    
    edges = count_edges(result)
    target = 76
    
    if edges >= target:
        return result

    # Deterministic Local Search (Hill Climbing)
    # Try moving each point to a nearby empty spot to increase edge count
    best_points = [p[:] for p in result]
    best_score = edges
    
    for _ in range(100): # Max iterations
        if time.time() > deadline - 5.0:
            break
        if best_score >= target:
            break
        
        improved = False
        for i in range(32):
            a, b = best_points[i]
            # Try all moves within distance 2
            for da in range(-2, 3):
                for db in range(-2, 3):
                    if da == 0 and db == 0:
                        continue
                    new_a, new_b = a + da, b + db
                    if abs(new_a) > 64 or abs(new_b) > 64:
                        continue
                    
                    # Check if occupied
                    occupied = False
                    for j, p in enumerate(best_points):
                        if j != i and p[0] == new_a and p[1] == new_b:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    # Check distance >= 1
                    valid = True
                    for j, p in enumerate(best_points):
                        if j == i:
                            continue
                        if dist_sq([new_a, new_b], p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    old_p = best_points[i]
                    best_points[i] = [new_a, new_b]
                    new_score = count_edges(best_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                        # Keep the move
                    else:
                        # Revert
                        best_points[i] = old_p
        
        if not improved:
            break

    if best_score > edges:
        result = best_points
    
    return result
