"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Generate hexagon of radius 3 (37 points)
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a - b)) <= 3:
                points.append((a, b))
    
    # We need to remove 5 points to get 32 points.
    # Try all combinations of 5 points to remove and pick the one that maximizes edges.
    
    def count_edges(point_set):
        """Count number of edges (pairs at distance 1) in point_set."""
        edges = 0
        for p in point_set:
            a, b = p
            for da, db in [(1, 0), (0, 1), (1, -1)]:  # Only 3 directions to avoid double counting
                q = (a + da, b + db)
                if q in point_set:
                    edges += 1
        return edges
    
    best_count = 0
    best_removed = None
    
    for removed_combo in combinations(points, 5):
        if time.time() > deadline:
            break
        remaining = set(points) - set(removed_combo)
        count = count_edges(remaining)
        if count > best_count:
            best_count = count
            best_removed = removed_combo
            if best_count == 76:
                break  # Can't do better
    
    # If we found a good solution, use it
    if best_removed is not None:
        current_points = set(points) - set(best_removed)
    else:
        # Fallback: greedy removal
        current_points = set(points)
        for _ in range(5):
            best_point = None
            best_loss = float('inf')
            for p in current_points:
                a, b = p
                deg = 0
                for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                    q = (a + da, b + db)
                    if q in current_points:
                        deg += 1
                if deg < best_loss:
                    best_loss = deg
                    best_point = p
            if best_point is not None:
                current_points.remove(best_point)
    
    # Convert back to list of lists
    result = [[a, b] for a, b in current_points]
    return result
# EVOLVE-BLOCK-END
