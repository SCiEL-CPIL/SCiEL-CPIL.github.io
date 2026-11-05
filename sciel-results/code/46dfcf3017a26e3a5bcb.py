"""Rewritten program: local search to minimize ordinary lines for 12 integer points."""
import time
import random
from math import gcd
from itertools import combinations

# Deadline guard: keep a safety margin before the 150s kill
_DEADLINE = time.time() + 135.0  # 15s safety margin


def _key(p):
    """Canonical key for a line through two points: (a, b, c) normalized."""
    x1, y1 = p[0]
    x2, y2 = p[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
    if g:
        a //= g
        b //= g
        c //= g
    # Canonical sign: make first non-zero positive
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_ordinary(points):
    """Count lines containing exactly 2 of the points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _key((points[i], points[j]))
            line_counts[k] = line_counts.get(k, 0) + 1
    return sum(1 for v in line_counts.values() if v == 2)


def _all_collinear(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    return (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0)


def _valid(points):
    """Check validity: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return not _all_collinear(points)


def _initial_points():
    """Start from a 3x4 grid (known good: 6 ordinary lines)."""
    pts = [[x, y] for x in range(4) for y in range(3)]
    return pts


def _random_neighbor(points, rng):
    """Generate a random neighbor by perturbing one point."""
    new_pts = [list(p) for p in points]
    i = rng.randrange(12)
    # Random small perturbation
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    if dx == 0 and dy == 0:
        dx = 1
    new_pts[i] = [new_pts[i][0] + dx, new_pts[i][1] + dy]
    return new_pts


def _random_restart(rng):
    """Generate a random valid starting configuration."""
    # Use a grid-like structure with some randomness
    base_x = rng.randint(-5, 5)
    base_y = rng.randint(-5, 5)
    sx = rng.randint(1, 4)
    sy = rng.randint(1, 4)
    cols = rng.randint(3, 5)
    rows = 12 // cols if 12 % cols == 0 else 4
    pts = []
    for r in range(rows):
        for c in range(cols):
            if len(pts) >= 12:
                break
            x = base_x + c * sx + rng.randint(-1, 1)
            y = base_y + r * sy + rng.randint(-1, 1)
            pts.append([x, y])
    # Fill remaining
    while len(pts) < 12:
        pts.append([rng.randint(-10, 10), rng.randint(-10, 10)])
    return pts


def construct():
    rng = random.Random(42)
    
    # Start with the known 3x4 grid
    best = _initial_points()
    best_score = _count_ordinary(best)
    
    # Try to improve via local search
    current = [list(p) for p in best]
    cur_score = best_score
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < _DEADLINE:
        # Generate a neighbor
        neighbor = _random_neighbor(current, rng)
        
        if not _valid(neighbor):
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart
                current = _random_restart(rng)
                if not _valid(current):
                    current = [list(p) for p in _initial_points()]
                cur_score = _count_ordinary(current)
                no_improve = 0
            continue
        
        n_score = _count_ordinary(neighbor)
        
        if n_score < cur_score:
            current = neighbor
            cur_score = n_score
            no_improve = 0
            if cur_score < best_score:
                best = [list(p) for p in current]
                best_score = cur_score
                if best_score <= 6:
                    break  # Can't do better than theoretical minimum
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Try a different restart strategy
                current = _random_restart(rng)
                if not _valid(current):
                    current = [list(p) for p in _initial_points()]
                cur_score = _count_ordinary(current)
                no_improve = 0
    
    # Final validation
    if not _valid(best):
        best = _initial_points()
    
    return best
