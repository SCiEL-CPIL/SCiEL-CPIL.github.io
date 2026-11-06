<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Use a set of normalized line representations
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through two points: a*x + b*y + c = 0
            # a = y2 - y1, b = x1 - x2, c = x2*y1 - x1*y2
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Normalize the line representation
            # Find GCD of a, b, c to normalize
            import math
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g == 0:
                g = 1
            a, b, c = a // g, b // g, c // g
            
            # Ensure consistent sign: make the first non-zero of (a,b,c) positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Count lines with exactly 2 points
    ordinary = sum(1 for count in line_counts.values() if count == 1)
    return ordinary

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin
    
    # The 3x4 grid is known to achieve the theoretical minimum of n/2 = 6 ordinary lines
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    
    # Verify the 3x4 grid
    ord_3x4 = count_ordinary_lines(grid_3x4)
    
    # Try a few other configurations to see if we can do better
    best_points = grid_3x4
    best_ordinary = ord_3x4
    
    # Try 2x6 grid
    if time.time() < deadline - 5:
        grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
        ord_2x6 = count_ordinary_lines(grid_2x6)
        if ord_2x6 < best_ordinary:
            best_points = grid_2x6
            best_ordinary = ord_2x6
    
    # Try 1x12 (all collinear - invalid, but check)
    # Not all collinear is required, so skip
    
    # Try a hexagonal-like pattern
    if time.time() < deadline - 5:
        # 3 rows of 4, shifted
        hex_pattern = [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0, 1], [1, 1], [2, 1], [3, 1],
            [0, 2], [1, 2], [2, 2], [3, 2]
        ]
        # This is the same as 3x4 grid, skip
    
    # Try other small grids
    if time.time() < deadline - 5:
        # 4x3 grid (same as 3x4)
        pass
    
    # Try a configuration with more collinear points
    if time.time() < deadline - 5:
        # Points on two lines: 6 points on one line, 6 on another
        # But they share an intersection point... let's think
        # Line 1: y=0, points (0,0) to (5,0)
        # Line 2: x=0, points (0,0) to (0,5)
        # That's 11 points with (0,0) shared, need one more
        two_lines = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 6)]
        # That's 11 points, add one more
        two_lines.append([5, 5])
        ord_two = count_ordinary_lines(two_lines)
        if ord_two < best_ordinary:
            best_points = two_lines
            best_ordinary = ord_two
    
    # The theoretical minimum is n/2 = 6, so we can't do better than 6
    # Return the best found
    return [[p[0], p[1]] for p in best_points]

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    
    # Canonical sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines (lines with exactly 2 points) efficiently."""
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    
    return sum(1 for cnt in line_counts.values() if cnt == 1)

def _is_valid(points):
    """Check if points are valid: 12 distinct, not all collinear, coords in range."""
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(tuple(p))
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    
    # Check not all collinear
    if len(points) >= 3:
        p0, p1, p2 = points[0], points[1], points[2]
        # Check if all points are on line through p0, p1
        a = p1[1] - p0[1]
        b = p0[0] - p1[0]
        c = p1[0]*p0[1] - p0[0]*p1[1]
        all_collinear = True
        for p in points:
            if a*p[0] + b*p[1] + c != 0:
                all_collinear = False
                break
        if all_collinear:
            return False
    
    return True

def _try_structured_configs():
    """Try known structured configurations and return the best."""
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4x3 grid (transpose of 3x4, same count)
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Try a configuration with points on few lines
    # 6 points on y=0, 5 on y=1, 1 on y=2
    candidates.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(5)] + [[2, 2]])
    
    # Hesse-like: try to create more 3-point lines
    # Points forming a triangular lattice
    tri = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    candidates.append(tri)
    
    # Shifted rows to create more diagonal 3-point lines
    shifted = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [1, 1], [2, 1], [3, 1], [4, 1],
        [2, 2], [3, 2], [4, 2], [5, 2]
    ]
    candidates.append(shifted)
    
    # Another shifted pattern
    shifted2 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [1, 2], [2, 2], [3, 2], [4, 2]
    ]
    candidates.append(shifted2)
    
    best_pts = None
    best_ord = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        ord_count = _count_ordinary_fast(pts)
        if ord_count < best_ord:
            best_ord = ord_count
            best_pts = [list(p) for p in pts]
    
    return best_pts, best_ord

def _local_search(points, deadline, max_iters=50000):
    """Local search: try moving individual points to reduce ordinary lines."""
    best_pts = [list(p) for p in points]
    best_ord = _count_ordinary_fast(best_pts)
    
    rng = random.Random(42)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        
        # Pick a random point to move
        idx = rng.randint(0, 11)
        old_x, old_y = best_pts[idx]
        
        # Try nearby positions
        moved = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                # Create new point set
                test_pts = [list(p) for p in best_pts]
                test_pts[idx] = [new_x, new_y]
                
                if not _is_valid(test_pts):
                    continue
                
                test_ord = _count_ordinary_fast(test_pts)
                if test_ord < best_ord:
                    best_pts = test_pts
                    best_ord = test_ord
                    old_x, old_y = new_x, new_y
                    moved = True
                    break
            if moved:
                break
        
        # Occasionally try a larger jump
        if iters % 100 == 0:
            idx = rng.randint(0, 11)
            old_x, old_y = best_pts[idx]
            for _ in range(5):
                new_x = old_x + rng.randint(-10, 10)
                new_y = old_y + rng.randint(-10, 10)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                test_pts = [list(p) for p in best_pts]
                test_pts[idx] = [new_x, new_y]
                
                if not _is_valid(test_pts):
                    continue
                
                test_ord = _count_ordinary_fast(test_pts)
                if test_ord < best_ord:
                    best_pts = test_pts
                    best_ord = test_ord
                    break
    
    return best_pts, best_ord

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin
    
    # First try structured configurations
    best_pts, best_ord = _try_structured_configs()
    
    if best_pts is None:
        # Fallback to 3x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_ord = _count_ordinary_fast(best_pts)
    
    # Local search to improve
    if time.time() < deadline - 5:
        improved_pts, improved_ord = _local_search(best_pts, deadline - 5, max_iters=100000)
        if improved_ord < best_ord:
            best_pts = improved_pts
            best_ord = improved_ord
    
    # Try multiple random restarts if time allows
    if time.time() < deadline - 10:
        rng = random.Random(123)
        for restart in range(20):
            if time.time() >= deadline - 5:
                break
            
            # Generate a random starting configuration
            rand_pts = []
            used = set()
            while len(rand_pts) < 12:
                x = rng.randint(-5, 5)
                y = rng.randint(-5, 5)
                if (x, y) not in used:
                    used.add((x, y))
                    rand_pts.append([x, y])
            
            if not _is_valid(rand_pts):
                continue
            
            rand_ord = _count_ordinary_fast(rand_pts)
            if rand_ord < best_ord:
                best_pts = rand_pts
                best_ord = rand_ord
            
            # Local search from this start
            if time.time() < deadline - 10:
                loc_pts, loc_ord = _local_search(rand_pts, deadline - 10, max_iters=20000)
                if loc_ord < best_ord:
                    best_pts = loc_pts
                    best_ord = loc_ord
    
    return [[p[0], p[1]] for p in best_pts]

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
