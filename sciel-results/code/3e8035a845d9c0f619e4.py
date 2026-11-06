import time
import random
import itertools
from math import gcd

_START_TIME = time.time()
_DEADLINE = 110  # safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _line_key(p1, p2):
    """Return normalized line key (a,b,c) for line through p1,p2: ax+by=c"""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    # Normalize
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    # Canonical sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _build_lines(points):
    """Build dict: line_key -> list of point indices on that line"""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = []
            lines[key].append(i)
            lines[key].append(j)
    # Deduplicate indices per line
    for key in lines:
        lines[key] = sorted(set(lines[key]))
    return lines

def _check_no_four_collinear(points):
    """Check no four points are collinear"""
    lines = _build_lines(points)
    for key, idxs in lines.items():
        if len(idxs) >= 4:
            return False
    return True

def _count_triples(points):
    """Count number of collinear triples"""
    lines = _build_lines(points)
    count = 0
    for key, idxs in lines.items():
        if len(idxs) >= 3:
            count += len(idxs) * (len(idxs) - 1) * (len(idxs) - 2) // 6
    return count

def _max_no3collinear_upper_bound(points, triples_set):
    """
    Fast upper bound on largest subset with no 3 collinear.
    Uses greedy: repeatedly remove a point that's in the most remaining triples.
    Returns the size of the remaining set (an upper bound on the true max).
    Actually this gives a LOWER bound on what we can achieve, but we want upper bound.
    
    Better: use the fact that if we have T triples, and we pick k points, 
    the number of triples among them is at most C(k,3).
    But that's weak. Let's use a greedy removal to find a large no-3-collinear subset,
    which gives a lower bound. For upper bound, we can use: 
    if every point is in at least d triples, then... hmm.
    
    Actually for search purposes, let's just compute the exact max no-3-collinear 
    for small candidate sets, and use triple count as a proxy during search.
    """
    pass

def _exact_max_no3collinear(points):
    """
    Exact branch and bound for largest subset with no 3 collinear.
    For 20 points this can be slow, so we use a heuristic + limited B&B.
    """
    n = len(points)
    if n <= 1:
        return n
    
    # Precompute: for each pair, which lines they're on, and for each point, 
    # which other points form a triple with it
    # triple[i] = set of (j,k) such that i,j,k are collinear
    lines = _build_lines(points)
    # For each point, the set of lines it's on
    point_lines = [set() for _ in range(n)]
    for key, idxs in lines.items():
        for idx in idxs:
            point_lines[idx].add(key)
    
    # triple_conflict[i] = set of j such that i and j are on a line with >=1 other point
    # Actually: conflict[i][j] = True if i,j are on some line with a third point
    conflict = [set() for _ in range(n)]
    for key, idxs in lines.items():
        if len(idxs) >= 3:
            for a in range(len(idxs)):
                for b in range(a+1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    conflict[i].add(j)
                    conflict[j].add(i)
    
    # Greedy: build a large no-3-collinear subset
    # Start with all points, remove points that are in conflicts
    # Better greedy: sort by conflict degree, remove high-degree points first
    # Then try to add back
    
    # Use a simple greedy: pick points one by one, maintaining no-3-collinear property
    # A point can be added if it doesn't form a triple with any two already-chosen points
    chosen = []
    chosen_set = set()
    remaining = list(range(n))
    remaining.sort(key=lambda i: -len(conflict[i]))
    
    for i in remaining:
        if i in chosen_set:
            continue
        # Check if adding i creates a triple with any two in chosen
        ok = True
        for j in chosen:
            if i in conflict[j]:
                # Check if there's a k in chosen such that i,j,k collinear
                # conflict[j] contains i means j and i share a line with >=1 other point
                # Need to check specifically
                pass
        # More careful check
        ok = True
        for j in chosen:
            for k in chosen:
                if j == k:
                    continue
                # Check if i, j, k are collinear
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(i)
            chosen_set.add(i)
    
    greedy_size = len(chosen)
    
    # Try to improve: for each point not in chosen, see if we can swap
    # For now, return greedy as a lower bound
    # For upper bound, we need B&B which is expensive
    
    return greedy_size

def _fast_score(points):
    """
    Fast approximate score: count triples and use that as a proxy.
    More triples -> smaller max no-3-collinear subset (roughly).
    For search, use triple count as objective to minimize.
    """
    return _count_triples(points)

def _try_local_search(points, lines, current_triples, max_iters=5000):
    """
    Local search: try moving one point to a nearby position to reduce triple count.
    Returns improved points or None.
    """
    n = len(points)
    best = list(points)
    best_triples = current_triples
    no_improve = 0
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        if no_improve > 100:
            break
        
        # Pick a random point to move
        i = random.randint(0, n - 1)
        old_x, old_y = best[i]
        
        # Try nearby positions
        found = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old_x + dx, old_y + dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                new_pt = [nx, ny]
                # Check distinctness
                if any(new_pt == best[j] for j in range(n) if j != i):
                    continue
                # Build new points
                new_points = [list(p) for p in best]
                new_points[i] = new_pt
                # Quick check: no four collinear
                if not _check_no_four_collinear(new_points):
                    continue
                # Count triples
                nt = _count_triples(new_points)
                if nt < best_triples:
                    best = new_points
                    best_triples = nt
                    no_improve = 0
                    found = True
                    break
            if found:
                break
        if not found:
            no_improve += 1
    
    return best, best_triples

def construct():
    random.seed(42)
    
    # Start with a grid-like configuration
    # 5x5 grid = 25 points, pick 20
    # Use a subset of a 7x7 grid for more structure
    
    # Strategy: start with points from a grid, then local search
    # Grid points: (i, j) for i,j in range(0, 7) -> 49 points
    # We need 20 with no four collinear
    
    # Actually, a full 5x5 grid has many 4-collinear (rows, cols, diagonals)
    # We need to pick a subset with no four collinear
    
    # Let's start with a known good structure: 
    # Points on a "bent" grid or use a finite geometry inspired approach
    
    # Simple approach: start with random points, then local search to maximize triples
    # while maintaining no-four-collinear
    
    best_points = None
    best_triples = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Generate initial points: mix of grid and random
        points = []
        used = set()
        
        # Start with a grid subset
        grid_size = 6
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= 20:
                    break
                pt = [i * 15 - 37, j * 15 - 37]  # spread out
                if abs(pt[0]) <= 100 and abs(pt[1]) <= 100:
                    if (pt[0], pt[1]) not in used:
                        points.append(pt)
                        used.add((pt[0], pt[1]))
        
        # Fill remaining with random points
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                points.append([x, y])
                used.add((x, y))
        
        # Ensure no four collinear (fix if needed)
        valid = False
        for attempt in range(100):
            if _check_no_four_collinear(points):
                valid = True
                break
            # Move a random point
            i = random.randint(0, 19)
            old_x, old_y = points[i]
            for _ in range(50):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) in used and [nx, ny] != points[i]:
                    continue
                if (nx, ny) == (old_x, old_y):
                    continue
                used.discard((old_x, old_y))
                used.add((nx, ny))
                points[i] = [nx, ny]
                if _check_no_four_collinear(points):
                    valid = True
                    break
            if valid:
                break
        
        if not valid:
            continue
        
        # Local search to maximize triples (minimize score)
        points, triples = _try_local_search(points, None, _count_triples(points), max_iters=2000)
        
        if triples < best_triples:
            best_triples = triples
            best_points = [list(p) for p in points]
    
    # If we found something, do a final polish
    if best_points is not None:
        # Additional local search passes
        for _ in range(3):
            if _elapsed() > _DEADLINE - 2:
                break
            best_points, best_triples = _try_local_search(best_points, None, best_triples, max_iters=5000)
    
    if best_points is None:
        # Fallback: return parabola (score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]
    
    # Final validation
    assert len(best_points) == 20
    assert all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in best_points)
    assert len(set(tuple(p) for p in best_points)) == 20
    assert _check_no_four_collinear(best_points)
    
    return best_points
