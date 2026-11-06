"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Reduce by gcd
    from math import gcd
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: y = m(x - x1) + y1, or x = x1 if vertical
    # Use (a, b, c) for ax + by = c where a=dy, b=-dx, c = dy*x1 - dx*y1
    a = dy
    b = -dx
    c = a * x1 + b * y1
    # Normalize sign: make (a, b) canonical
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    # Reduce by gcd of (a, b, c)
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _count_exact_four(points):
    """Count lines containing exactly 4 of the given points. Returns (count, has_five)."""
    n = len(points)
    if n < 4:
        return 0, False
    line_points = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_points[key].append((i, j))
    
    # For each line, count how many of the 16 points lie on it
    # We need to be careful: multiple pairs on the same line
    # Better approach: for each line key, collect all point indices on it
    
    line_to_indices = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_to_indices[key].add(i)
                line_to_indices[key].add(j)
    
    count = 0
    has_five = False
    for key, indices in line_to_indices.items():
        cnt = len(indices)
        if cnt == 4:
            count += 1
        elif cnt >= 5:
            has_five = True
    return count, has_five

def _is_valid(points):
    """Check: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, has_five = _count_exact_four(points)
    return not has_five

def _score(points):
    """Return (count, valid)."""
    if not _is_valid(points):
        return 0, False
    count, _ = _count_exact_four(points)
    return count, True

def _local_search(points, deadline):
    """Swap-based local search to improve the configuration."""
    best = list(points)
    best_score, _ = _score(best)
    
    n = 16
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                # Try swapping point i to a new position and point j to a new position
                # Simpler: try moving one point at a time
                pass
        # Try single-point moves
        for i in range(n):
            if time.time() >= deadline:
                break
            x0, y0 = best[i]
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check if new position is already occupied
                    if [nx, ny] in best:
                        continue
                    trial = list(best)
                    trial[i] = [nx, ny]
                    s, valid = _score(trial)
                    if valid and s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        x0, y0 = nx, ny  # update for further moves from new pos
                        break
                if time.time() >= deadline:
                    break
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s limit
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best, best_score = _local_search(points, deadline)
    
    # Try a few different starting configurations
    # Configuration 2: shifted grid
    points2 = [[x + 1, y + 1] for x in range(4) for y in range(4)]
    b2, s2 = _local_search(points2, deadline)
    if s2 > best_score:
        best, best_score = b2, s2
    
    # Configuration 3: non-square grid (e.g., 4x4 but with wider spacing)
    points3 = [[x * 2, y * 2] for x in range(4) for y in range(4)]
    b3, s3 = _local_search(points3, deadline)
    if s3 > best_score:
        best, best_score = b3, s3
    
    # Configuration 4: 4x4 grid with one point moved
    points4 = [[x, y] for x in range(4) for y in range(4)]
    points4[0] = [5, 5]
    b4, s4 = _local_search(points4, deadline)
    if s4 > best_score:
        best, best_score = b4, s4
    
    # Final verification
    final_score, valid = _score(best)
    if not valid or final_score < best_score:
        # Fallback to original grid
        best = [[x, y] for x in range(4) for y in range(4)]
    
    return best
# EVOLVE-BLOCK-END
