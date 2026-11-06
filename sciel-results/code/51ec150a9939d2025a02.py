"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import gcd

_start_time = time.time()
TIME_LIMIT = 140  # Leave 10s margin before 150s kill

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = {}  # line_key -> count of points on it
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            # Normalize
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Ensure consistent sign: make first non-zero positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
    # Count lines with exactly 2 points
    ordinary = sum(1 for cnt in lines.values() if cnt == 2)
    return ordinary

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        pt = (p[0], p[1])
        if pt in seen:
            return False
        seen.add(pt)
    # Check not all collinear
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    cross = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
    if cross == 0:
        # Check if all are collinear
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x1-x0)*(yi-y0) - (y1-y0)*(xi-x0)
            if cross_i != 0:
                return True
        return False
    return True

def _try_grid():
    """3x4 grid."""
    points = [[x, y] for x in range(4) for y in range(3)]
    return points

def _try_pappus_like():
    """Try a Pappus-inspired configuration for 12 points."""
    # Use points on two lines plus some structure
    # Actually, let's try a 2x6 arrangement or other
    pass

def _try_configurations():
    """Try several hand-crafted configurations and return the best."""
    candidates = []
    
    # 1. 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # 2. 3x4 grid (transposed)
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # 3. 2x6 grid
    pts3 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(pts3)
    
    # 4. 6x2 grid
    pts4 = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(pts4)
    
    # 5. Two concentric hexagons
    hex1 = [(2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1)]
    hex2 = [(4,0),(2,2),(-2,2),(-4,0),(-2,-2),(2,-2)]
    pts5 = [list(p) for p in hex1 + hex2]
    candidates.append(pts5)
    
    # 6. 4 points on each of 3 concurrent lines (excluding origin)
    pts6 = [
        [-2,0],[-1,0],[1,0],[2,0],
        [0,-2],[0,-1],[0,1],[0,2],
        [-2,-2],[-1,-1],[1,1],[2,2]
    ]
    candidates.append(pts6)
    
    # 7. 3 lines with 4 points each, not concurrent
    pts7 = [
        [4,0],[5,0],[6,0],[7,0],
        [4,1],[5,1],[6,1],[7,1],
        [0,0],[0,1],[0,2],[0,3]
    ]
    candidates.append(pts7)
    
    # 8. Points on a conic: try (i, i^2) for i in range, plus strategic additions
    #    For 12 points on a parabola, we need to add points that create more collinearities
    parabola = [[i, i*i] for i in range(6)]
    # Add 6 more points that lie on lines through pairs of parabola points
    # Line through (0,0) and (1,1): y=x, add (2,2),(3,3),(4,4),(5,5) but these are on parabola? No, (2,4),(3,9),(4,16),(5,25)
    # Actually let's try a different approach: use points that create many 3-point lines
    # Known good configuration: 12 points with 6 ordinary lines
    # Try: 4 points on 3 parallel lines, arranged to maximize cross-collinearities
    # Line y=0: (0,0),(1,0),(2,0),(3,0)
    # Line y=1: (0,1),(1,1),(2,1),(3,1)  
    # Line y=2: (0,2),(1,2),(2,2),(3,2)
    # This is 4x3 grid. Let's try shifting to create more diagonal collinearities
    pts8 = [
        [0,0],[2,0],[4,0],[6,0],
        [0,1],[2,1],[4,1],[6,1],
        [0,2],[2,2],[4,2],[6,2]
    ]
    candidates.append(pts8)
    
    # 9. Try a configuration with more symmetry
    #    12 points: 4 on x-axis, 4 on y-axis, 4 on y=x line (excluding origin)
    pts9 = [
        [-3,0],[-1,0],[1,0],[3,0],
        [0,-3],[0,-1],[0,1],[0,3],
        [-2,-2],[0,0],[2,2],[4,4]  # Wait, origin is already used? No, but let's avoid duplicates
    ]
    # Fix: remove origin from third line, use different points
    pts9 = [
        [-3,0],[-1,0],[1,0],[3,0],
        [0,-3],[0,-1],[0,1],[0,3],
        [-2,-2],[1,1],[2,2],[4,4]
    ]
    candidates.append(pts9)
    
    # 10. Try a "grid with gaps" to create more diagonals
    pts10 = [
        [0,0],[1,0],[3,0],[4,0],
        [0,1],[1,1],[3,1],[4,1],
        [0,2],[1,2],[3,2],[4,2]
    ]
    candidates.append(pts10)
    
    # 11. Try points that form multiple 3-point lines
    #    Based on known results: minimum for 12 points is 6 ordinary lines
    #    Try a configuration inspired by the orchard-planting problem
    pts11 = [
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[2,1],[3,1],[4,1],
        [0,2],[1,2],[3,2],[4,2]
    ]
    candidates.append(pts11)
    
    # 12. Try a more compact configuration
    pts12 = [
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[1,1],[2,1],[4,1],
        [0,2],[1,2],[3,2],[4,2]
    ]
    candidates.append(pts12)
    
    best_pts = None
    best_count = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        cnt = _count_ordinary(pts)
        if cnt < best_count:
            best_count = cnt
            best_pts = [p[:] for p in pts]
    
    return best_pts, best_count

def _local_search(points, max_iters=5000):
    """Try to improve by moving individual points."""
    if not _is_valid(points):
        return points, _count_ordinary(points)
    
    current_pts = [p[:] for p in points]
    current_count = _count_ordinary(current_pts)
    
    # Search space for new points: small integer grid
    # Try moving one point at a time to a nearby location
    search_range = 10
    base_x = sum(p[0] for p in current_pts) // 12
    base_y = sum(p[1] for p in current_pts) // 12
    
    # Maintain a set of occupied points for O(1) lookup
    occupied = set((p[0], p[1]) for p in current_pts)
    
    for iteration in range(max_iters):
        if time.time() - _start_time > TIME_LIMIT - 5:
            break
        improved = False
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Generate a smaller set of candidates to ensure speed
        # Random nearby points
        candidates = []
        for _ in range(30):
            dx = random.randint(-search_range, search_range)
            dy = random.randint(-search_range, search_range)
            nx = old_x + dx
            ny = old_y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.append((nx, ny))
        
        for nx, ny in candidates:
            if time.time() - _start_time > TIME_LIMIT - 5:
                break
            # Check if this position is already used
            if (nx, ny) in occupied:
                continue
            
            # Remove old, add new
            occupied.remove((old_x, old_y))
            occupied.add((nx, ny))
            
            new_pts = [p[:] for p in current_pts]
            new_pts[idx] = [nx, ny]
            
            if not _is_valid(new_pts):
                occupied.remove((nx, ny))
                occupied.add((old_x, old_y))
                continue
                
            cnt = _count_ordinary(new_pts)
            if cnt < current_count:
                current_pts = new_pts
                current_count = cnt
                improved = True
                break
            else:
                # Revert
                occupied.remove((nx, ny))
                occupied.add((old_x, old_y))
        
        if not improved:
            pass
    
    return current_pts, current_count

def construct():
    global _start_time
    _start_time = time.time()
    
    random.seed(42)
    
    # First try hand-crafted configurations
    best_pts, best_count = _try_configurations()
    
    if best_pts is None:
        # Fallback: 4x3 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_count = _count_ordinary(best_pts)
    
    # Local search to improve
    if time.time() - _start_time < TIME_LIMIT - 5:
        best_pts, best_count = _local_search(best_pts, max_iters=2000)
    
    # Final validation
    if not _is_valid(best_pts):
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    
    return best_pts
# EVOLVE-BLOCK-END
