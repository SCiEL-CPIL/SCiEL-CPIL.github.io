"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 140  # seconds, safety margin before 150s kill
_START_TIME = None

def _line_key(i, j, points):
    """Compute canonical line key for pair (i,j)."""
    x1, y1 = points[i]
    x2, y2 = points[j]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)
    elif dy == 0:
        return (0, 1, y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        if dy_n < 0 or (dy_n == 0 and -dx_n < 0):
            dy_n = -dy_n
            dx_n = -dx_n
            c = -c
        return (dy_n, -dx_n, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(i, j, points)
            line_count[key] = line_count.get(key, 0) + 1
    # Each ordinary line has exactly 1 pair, non-ordinary with k points has C(k,2) pairs
    # Number of ordinary lines = number of keys with count == 1
    ordinary = 0
    for key, cnt in line_count.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _structured_hexagon():
    """Construct 12 points from a hexagon + hexagram intersection pattern.
    Use regular hexagon vertices and their diagonal intersection points."""
    # Regular hexagon with radius R centered at origin
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For integer coords, use a scaled version
    # Hexagon vertices: (R,0), (R/2, R*sqrt(3)/2), ... - not integer
    # Instead, use a combinatorial construction:
    # 3 lines with 4 points each, arranged in a triangular pattern
    
    # Construction: 12 points as union of 3 lines, each with 4 points
    # Line 1: y = 0, x = 0, 1, 2, 3
    # Line 2: y = 1, x = 0, 1, 2, 3  
    # Line 3: y = 2, x = 0, 1, 2, 3
    # This is just a 4x3 grid = 12 points
    
    # Better: use a configuration with more collinearities
    # 4 points on each of 3 non-parallel lines, with cross-collinearities
    
    # Try: points at integer coordinates forming a pattern
    # Use the known optimal-ish configuration for small n
    # For 12 points, try: 2 rows of 6, but offset to create cross-lines
    
    # Configuration A: 3x4 grid
    pts_a = [[x, y] for x in range(4) for y in range(3)]
    
    # Configuration B: hexagon-like with integer coords
    # Use points: (0,0), (2,0), (4,0), (1,1), (3,1), (5,1), 
    #             (0,2), (2,2), (4,2), (1,3), (3,3), (5,3)
    pts_b = [[0,0],[2,0],[4,0],[1,1],[3,1],[5,1],
             [0,2],[2,2],[4,2],[1,3],[3,3],[5,3]]
    
    # Configuration C: more structured
    # 4 points on y=0: (0,0),(1,0),(2,0),(3,0)
    # 4 points on y=1: (0,1),(1,1),(2,1),(3,1)
    # 4 points on y=2: (0,2),(1,2),(2,2),(3,2)
    # Plus diagonals create extra collinearities
    
    return [pts_a, pts_b]

def _local_search(points, max_iter=5000):
    """Perform local search to minimize ordinary lines."""
    global _START_TIME
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)
    
    for it in range(max_iter):
        if time.time() - _START_TIME > DEADLINE - 5:
            break
        
        # Randomly pick a point to move
        i = random.randint(0, len(best) - 1)
        old_x, old_y = best[i]
        
        # Try a few nearby random locations
        accepted = False
        for _ in range(10):
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            if new_x == old_x and new_y == old_y:
                continue
            # Check distinctness
            valid = True
            for j, p in enumerate(best):
                if j != i and p[0] == new_x and p[1] == new_y:
                    valid = False
                    break
            if not valid:
                continue
            
            best[i] = [new_x, new_y]
            count = _count_ordinary(best)
            
            if count < best_count:
                best_count = count
                accepted = True
                break
            else:
                best[i] = [old_x, old_y]
    
    return best, best_count

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Try multiple structured initial configurations
    configs = _structured_hexagon()
    
    best_points = None
    best_count = float('inf')
    
    for start_pts in configs:
        if time.time() - _START_TIME > DEADLINE - 10:
            break
        
        start = [p[:] for p in start_pts]
        result, count = _local_search(start, max_iter=3000)
        
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in result]
    
    # If no good result, use fallback grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_count = _count_ordinary(best_points)
    
    # Final validation
    seen = set()
    for p in best_points:
        if tuple(p) in seen:
            return [[x, y] for x in range(4) for y in range(3)]
        seen.add(tuple(p))
    
    # Check not all collinear
    if len(best_points) >= 3:
        x1, y1 = best_points[0]
        x2, y2 = best_points[1]
        x3, y3 = best_points[2]
        if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
            all_col = True
            for p in best_points[3:]:
                if (x2 - x1) * (p[1] - y1) != (y2 - y1) * (p[0] - x1):
                    all_col = False
                    break
            if all_col:
                return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
# EVOLVE-BLOCK-END
