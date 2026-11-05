"""Rewritten: line-packing construction + local search + exact branch-and-bound."""
import time
import itertools
from fractions import Fraction

_DEADLINE = None
_TIME_LIMIT = 150.0

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - 10.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _valid(points):
    """Check: 20 distinct points, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check no four collinear
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = _gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            ux, uy = dx // g, dy // g
            count = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                ddx = points[k][0] - points[i][0]
                ddy = points[k][1] - points[i][1]
                if ddx * uy == ddy * ux:
                    count += 1
            if count >= 2:
                return False
    return True

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(i, j, points):
    """Return normalized (dx, dy) for line through points[i], points[j]."""
    dx = points[j][0] - points[i][0]
    dy = points[j][1] - points[i][1]
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0)
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy)

def _all_lines(points):
    """Return dict: (line_key, offset) -> list of point indices on that line.
    We use (ux, uy, c) where c = ux*y - uy*x (the line equation ux*y - uy*x = c)."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = _gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            ux, uy = dx // g, dy // g
            if ux < 0 or (ux == 0 and uy < 0):
                ux, uy = -ux, -uy
            # Line: ux*(y - y_i) = uy*(x - x_i) => ux*y - uy*x = ux*y_i - uy*x_i
            c = ux * points[i][1] - uy * points[i][0]
            key = (ux, uy, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _count_triples(points):
    """Count number of 3-point collinear subsets."""
    lines = _all_lines(points)
    count = 0
    for pts in lines.values():
        if len(pts) >= 3:
            count += 1  # Each line with >=3 points contributes C(len,3) triples
            # Actually we want C(len,3)
            count -= 1
            count += _comb(len(pts), 3)
    return count

def _comb(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def _build_initial():
    """5x4 grid with one point perturbed to break 4-collinear lines."""
    # 5 columns x 4 rows
    cols = [-40, -20, 0, 20, 40]
    rows = [-30, -10, 10, 30]
    points = []
    for r in rows:
        for c in cols:
            points.append([c, r])
    # Point at index 12 is (40, 30) - top right corner
    # Move it to break the diagonal and row/col 4-collinear
    # The main diagonal from (-40,-30) to (40,30) has 5 points: indices 0,6,11,17,23? Let me recompute.
    # Actually with 5x4 grid, diagonals: 
    #   slope 30/20=3/2: from (-40,-30) going up-right: (-40,-30),(-20,0),(0,10),(20,20)... not integer in grid
    # Let me just use a simpler approach: 5x4 grid, move one point slightly.
    # The row y=30 has 5 points -> 4-collinear! Bad.
    # So I need a different construction.
    
    # Better: use a 5x4 grid but with non-uniform spacing to avoid 4+ on any line.
    # Or: use the known "no 4 collinear" grid trick.
    
    # Let me use: x in {0,1,3,7,15}, y in {0,1,2,4} scaled and shifted
    # Actually simplest: take 20 points and verify no 4 collinear, then add 3-point lines.
    
    # New approach: Start with a configuration known to have no 4 collinear.
    # Use points on y = x^2 but spread out, then add a few off-parabola to create 3-lines.
    # Parabola y=x^2 for x in -10..9 gives 20 points, no 3 collinear. Score=20.
    # To reduce score, I need to create 3-point lines.
    # Replace some parabola points with points that create 3-collinear sets.
    
    # Strategy: Take 10 points on parabola, 10 points arranged to create many 3-lines.
    # Or: Take a 4x5 grid (20 points) and verify no 4 collinear.
    # A 4x5 grid: rows y=0,1,2,3; cols x=0,1,2,3,4. 
    # Row y=0: 5 points -> 4-collinear! Bad.
    
    # So I need irregular spacing. Let me use:
    # x values: 0, 1, 3, 7, 15 (5 values, no 4 in arithmetic progression)
    # y values: 0, 1, 3, 7 (4 values)
    # 5*4 = 20 points. Check for 4-collinear.
    
    xvals = [0, 1, 3, 7, 15]
    yvals = [0, 1, 3, 7]
    pts = []
    for y in yvals:
        for x in xvals:
            pts.append([x, y])
    
    # Check no 4 collinear
    if not _valid(pts):
        # Try different xvals
        xvals = [0, 1, 3, 7, 14]
        pts = []
        for y in yvals:
            for x in xvals:
                pts.append([x, y])
    
    if not _valid(pts):
        xvals = [0, 1, 4, 9, 16]
        pts = []
        for y in yvals:
            for x in xvals:
                pts.append([x, y])
    
    if not _valid(pts):
        # Fallback: parabola (valid but score=20)
        pts = [[x, x*x] for x in range(-10, 10)]
    
    # Scale to fit in [-100,100]
    max_x = max(abs(p[0]) for p in pts)
    max_y = max(abs(p[1]) for p in pts)
    scale = min(100 // max(1, max_x), 100 // max(1, max_y), 10)
    if scale < 1:
        scale = 1
    pts = [[p[0]*scale, p[1]*scale] for p in pts]
    
    return pts

def _local_search(points, max_iters=5000):
    """Try to improve by moving points to create more 3-collinear lines."""
    best = points[:]
    best_triples = _count_triples(best)
    n = len(best)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        improved = False
        # Try moving each point to a nearby location
        for i in range(n):
            if _time_left() < 5.0:
                break
            ox, oy = best[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    # Check distinctness
                    if any(best[j][0] == nx and best[j][1] == ny for j in range(n) if j != i):
                        continue
                    trial = best[:]
                    trial[i] = [nx, ny]
                    if not _valid(trial):
                        continue
                    t = _count_triples(trial)
                    if t > best_triples:
                        best = trial
                        best_triples = t
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_triples

def _max_3free_subset(points):
    """Exact branch and bound for largest subset with no 3 collinear."""
    n = len(points)
    
    # Precompute: for each pair (i,j), which other points are collinear with them
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            for k in range(n):
                if k == i or k == j:
                    continue
                ddx = points[k][0] - points[i][0]
                ddy = points[k][1] - points[i][1]
                if dx * ddy == dy * ddx:
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)
    
    # Find all triples (i,j,k) that are collinear
    triples = []
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in collinear_with[i][j]:
                if k > j:
                    t = (i, j, k)
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
    
    best_size = 0
    best_subset = None
    
    # Branch and bound
    # State: (available_indices, current_subset)
    # available: set of indices we can still add
    # We branch on adding or not adding each point
    
    # Better: use the standard approach for "maximum independent set in 3-uniform hypergraph"
    # where hyperedges are the collinear triples.
    
    # Represent available as a bitmask (20 bits)
    avail = (1 << n) - 1
    
    def bb(avail_mask, current_size, current_set):
        nonlocal best_size, best_subset
        if _time_left() < 3.0:
            return
        if avail_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_subset = current_set[:]
            return
        # Upper bound: current_size + popcount(avail_mask)
        ub = current_size + bin(avail_mask).count('1')
        if ub <= best_size:
            return
        
        # Pick the point with the most constraints (in the most triples)
        # Or pick the smallest index for simplicity
        # Better: pick a point that's in the most remaining triples
        # For speed, just pick the lowest available bit
        # Find lowest set bit
        lsb = avail_mask & (-avail_mask)
        idx = lsb.bit_length() - 1
        
        # Option 1: include idx
        # Check if including idx violates any triple
        # A triple is violated if all 3 points are in current_set
        # When we add idx, check if there's a triple (idx, a, b) where a,b in current_set
        conflict = False
        for t in triples:
            if idx in t:
                others = [x for x in t if x != idx]
                if all(x in current_set for x in others):
                    conflict = True
                    break
        if not conflict:
            new_set = current_set + [idx]
            bb(avail_mask ^ lsb, current_size + 1, new_set)
        
        # Option 2: exclude idx
        bb(avail_mask ^ lsb, current_size, current_set)
    
    bb(avail, 0, [])
    return best_size

def construct():
    _init_deadline()
    
    # Step 1: Build initial configuration
    points = _build_initial()
    
    # Step 2: Local search to maximize 3-collinear triples
    points, triples_count = _local_search(points)
    
    # Step 3: Compute exact score
    score = _max_3free_subset(points)
    
    return points

# Test
if __name__ == "__main__":
    pts = construct()
    print(f"Points: {pts}")
    print(f"Valid: {_valid(pts)}")
    print(f"Triples: {_count_triples(pts)}")
    print(f"Score: {_max_3free_subset(pts)}")
