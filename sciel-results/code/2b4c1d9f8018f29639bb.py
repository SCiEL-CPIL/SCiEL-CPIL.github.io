"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line (defined by a pair) to the set of points on it.
    # A line is ordinary if it contains exactly 2 points.
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Normalize the line representation: (A, B, C) for Ax + By + C = 0
            # with gcd(|A|,|B|,|C|)=1 and sign convention.
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(abs(A), abs(B))
            if g:
                A //= g
                B //= g
                C //= g
            else:
                # A and B both zero shouldn't happen for distinct points
                A, B, C = 0, 0, 0
            # Sign convention: make the first non-zero of (A, B, C) positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _local_search(points, max_iters=2000, timeout=10.0):
    """Improve a configuration by moving individual points to nearby integer positions."""
    start = time.time()
    best = list(points)
    best_score = _count_ordinary(best)
    n = len(best)
    
    for _ in range(max_iters):
        if time.time() - start > timeout:
            break
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = best[idx]
        # Try a few nearby positions
        improved = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = old_x + dx, old_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                # Check distinctness
                candidate = list(best)
                candidate[idx] = [new_x, new_y]
                # Check if new point duplicates another
                dup = False
                for j, (px, py) in enumerate(candidate):
                    if j != idx and px == new_x and py == new_y:
                        dup = True
                        break
                if dup:
                    continue
                # Check not all collinear
                if _all_collinear(candidate):
                    continue
                score = _count_ordinary(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return best, best_score


def _all_collinear(points):
    if len(points) < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
            return False
    return True


def construct():
    """Return 12 integer points minimizing ordinary lines."""
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s hard kill
    
    # Start with known good configurations and improve via local search.
    # Strategy: use a hexagonal/structured arrangement that tends to have fewer ordinary lines.
    
    # Candidate 1: Two parallel lines of 6 points each, offset to create many collinearities
    # Points on y=0: (0,0)..(5,0); Points on y=1: (0,1)..(5,1)
    # This is a 6x2 grid. Let's also try other structured starts.
    
    candidates = []
    
    # 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 6x2 grid
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 2x6 grid
    candidates.append([[0, y] for y in range(6)] + [[1, y] for y in range(6)])
    
    # Hexagon-like: 2 rows of 3, plus 2 more rows offset
    # Try a more spread out configuration
    candidates.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
    ])  # 4x3
    
    # Try a configuration with more diagonal structure
    # 3 points on each of 4 lines in a pencil-like arrangement
    candidates.append([
        [0, 0], [1, 1], [2, 2],  # diagonal y=x
        [0, 1], [1, 2], [2, 3],  # diagonal y=x+1
        [0, -1], [1, 0], [2, 1], # diagonal y=x-1
        [0, 2], [1, 3], [2, 4],  # diagonal y=x+2
    ])
    
    # A configuration based on the Hessian configuration idea (but simplified)
    # 3x3 grid plus 3 more points
    candidates.append([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2],
    ])
    
    # Random structured starts
    for _ in range(20):
        # Random points in a small box
        pts = []
        for _ in range(12):
            pts.append([random.randint(-5, 5), random.randint(-5, 5)])
        # Remove duplicates
        seen = set()
        unique_pts = []
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)
        if len(unique_pts) == 12 and not _all_collinear(unique_pts):
            candidates.append(unique_pts)
    
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Ensure 12 distinct points
        seen = set()
        unique = []
        for p in cand:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        if len(unique) != 12:
            continue
        if _all_collinear(unique):
            continue
        score = _count_ordinary(unique)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in unique]
    
    # Local search on the best candidate
    if best_points is not None:
        best_points, best_score = _local_search(best_points, max_iters=3000, timeout=110)
    
    # If we still have time, try local search on other candidates
    if best_points is not None and time.time() < deadline - 15:
        for cand in candidates:
            if time.time() > deadline - 10:
                break
            seen = set()
            unique = []
            for p in cand:
                key = (p[0], p[1])
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            if len(unique) != 12 or _all_collinear(unique):
                continue
            # Skip if it's the same as best
            if sorted(map(tuple, unique)) == sorted(map(tuple, best_points)):
                continue
            improved, sc = _local_search(unique, max_iters=1000, timeout=min(10, deadline - time.time() - 5))
            if sc < best_score:
                best_score = sc
                best_points = improved
    
    if best_points is None:
        # Fallback: 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
