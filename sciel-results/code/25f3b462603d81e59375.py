"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

_TIME_LIMIT = 115.0  # safety margin before 150s kill

def _normalize_direction(dx, dy):
    """Return a canonical (dx, dy) for a line direction."""
    if dx == 0 and dy == 0:
        return (0, 0)
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = gcd(abs(dx), abs(dy))
    return (dx // g, dy // g)

def _line_signature(p1, p2):
    """Return a canonical signature for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dxn, dyn = _normalize_direction(dx, dy)
    # offset: for direction (a, b), the line is b*x - a*y = c
    c = dyn * x1 - dxn * y1
    return (dxn, dyn, c)

def _count_exact4_and_check5(points):
    """Count lines with exactly 4 points; return (exact4_count, has_5)."""
    n = len(points)
    if n < 4:
        return 0, False
    line_points = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            sig = _line_signature(points[i], points[j])
            line_points[sig].append(i)
    exact4 = 0
    has5 = False
    for sig, idxs in line_points.items():
        # deduplicate indices (a pair defines the line; each point on the line appears multiple times)
        unique = set(idxs)
        cnt = len(unique)
        if cnt >= 5:
            has5 = True
        elif cnt == 4:
            exact4 += 1
    return exact4, has5

def _score(points):
    """Objective: exact-4 lines, but invalid if 5-collinear."""
    exact4, has5 = _count_exact4_and_check5(points)
    if has5:
        return -1000000 - exact4
    return exact4

def _generate_seeds():
    """Generate diverse structured seeds."""
    seeds = []
    # 4x4 grid
    seeds.append([[x, y] for x in range(4) for y in range(4)])
    # 4x4 grid shifted
    seeds.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    # 5x5 grid minus center (16 points)
    pts = []
    for x in range(5):
        for y in range(5):
            if not (x == 2 and y == 2):
                pts.append([x, y])
    seeds.append(pts)
    # 4x4 with one point moved to break symmetry slightly
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 5]
    seeds.append(pts)
    # Random near-grid
    random.seed(42)
    pts = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(3):
        idx = random.randrange(16)
        pts[idx] = [random.randint(-2, 6), random.randint(-2, 6)]
    seeds.append(pts)
    # Another random seed
    random.seed(123)
    pts = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(5):
        idx = random.randrange(16)
        pts[idx] = [random.randint(-3, 7), random.randint(-3, 7)]
    seeds.append(pts)
    return seeds

def _local_search(points, time_limit):
    """Greedy single-point local search."""
    points = [p[:] for p in points]
    current_score = _score(points)
    if current_score < 0:
        # try to fix 5-collinear by moving a point
        pass
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n = len(points)
        for i in range(n):
            if time.time() > time_limit:
                return points, _score(points)
            ox, oy = points[i]
            # Try moving point i to nearby positions
            best_new = None
            best_new_score = current_score
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # check distinctness
                    if any(points[j] == [nx, ny] for j in range(n) if j != i):
                        continue
                    new_points = [p[:] for p in points]
                    new_points[i] = [nx, ny]
                    s = _score(new_points)
                    if s > best_new_score:
                        best_new_score = s
                        best_new = [nx, ny]
            if best_new is not None:
                points[i] = best_new
                current_score = best_new_score
                improved = True
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    seeds = _generate_seeds()
    best_points = None
    best_score = -10**9
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        # Ensure all points are within bounds and distinct
        valid = True
        seen = set()
        for p in seed:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid = False
                break
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid or len(seed) != 16:
            continue
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in seed]
        # Local search
        if time.time() < deadline - 10:
            searched, s2 = _local_search(seed, deadline - 10)
            if s2 > best_score:
                best_score = s2
                best_points = [p[:] for p in searched]
    # Final check
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
    return best_points
# EVOLVE-BLOCK-END
