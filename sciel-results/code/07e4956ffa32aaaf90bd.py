"""
Improved program: two-line config + local search with faster scoring.
The two-line configuration is known to give a strong baseline.
We add a local search that tries to perturb individual points to improve the dominant area multiplicity.
"""
import random
import time
import numpy as np
from itertools import combinations
from collections import Counter

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85  # 5s safety margin before 90s kill


def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                d = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    cnt = Counter(areas)
    return max(cnt.values())


def _score_fast(points, prev_counter=None, prev_best=None):
    """
    Faster incremental scoring: if we have a previous Counter and best,
    we can recompute only the areas involving the changed point.
    For simplicity, we just recompute fully but use numpy vectorization where possible.
    """
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all doubled areas using vectorized ops
    # For each triple (i,j,k), area = |det(b-a, c-a)|
    # We'll compute for all triples using broadcasting but that's O(n^3) memory.
    # Instead, iterate but in a tighter loop.
    areas = []
    for i in range(n):
        a = pts[i]
        for j in range(i+1, n):
            b = pts[j]
            for k in range(j+1, n):
                c = pts[k]
                d = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    cnt = Counter(areas)
    return max(cnt.values())


def _two_line_config():
    """15 points on y=0 (x=0..14), 15 points on y=1 (x=0..14)."""
    points = []
    for x in range(15):
        points.append((x, 0))
        points.append((x, 1))
    return points


def _local_search(points, max_iters=50000, timeout=None):
    """
    Local search: try moving each point to a nearby position and keep if score improves.
    Use a temperature-like acceptance or just hill climbing.
    """
    if timeout is None:
        timeout = 30
    start = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    n = len(best_points)
    # Neighbor offsets
    offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    random.shuffle(offsets) if False else None  # deterministic for reproducibility; use fixed order
    # Actually, let's use a fixed set of small offsets
    offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(-2,0),(2,0),(0,-2),(0,2)]
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        if time.time() > start + timeout:
            break
        for i in range(n):
            if time.time() > start + timeout:
                break
            x, y = best_points[i]
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                # Check if already occupied
                if (nx, ny) in best_points:
                    continue
                # Try the move
                new_points = list(best_points)
                new_points[i] = (nx, ny)
                s = _score_fast(new_points)
                if s > best_score:
                    best_points = new_points
                    best_score = s
                    improved = True
                    iter_count += 1
                    break  # restart inner loop for this point
                iter_count += 1
    return best_points, best_score


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85
    
    # Start with two-line configuration
    best_points = _two_line_config()
    best_score = _score(best_points)
    
    # Local search from two-line config
    if time.time() < _DEADLINE - 10:
        best_points, best_score = _local_search(best_points, max_iters=30000, timeout=_DEADLINE - time.time() - 5)
    
    # Try 3x10 grid as another starting point
    grid310 = []
    for y in range(3):
        for x in range(10):
            grid310.append((x, y))
    g310_score = _score(grid310)
    if g310_score > best_score:
        best_points = grid310
        best_score = g310_score
    
    if time.time() < _DEADLINE - 5:
        g310_best, g310_best_score = _local_search(grid310, max_iters=20000, timeout=_DEADLINE - time.time() - 3)
        if g310_best_score > best_score:
            best_points = g310_best
            best_score = g310_best_score
    
    # Try 5x6 grid
    grid56 = []
    for y in range(5):
        for x in range(6):
            grid56.append((x, y))
    g56_score = _score(grid56)
    if g56_score > best_score:
        best_points = grid56
        best_score = g56_score
    
    if time.time() < _DEADLINE - 3:
        g56_best, g56_best_score = _local_search(grid56, max_iters=15000, timeout=_DEADLINE - time.time() - 2)
        if g56_best_score > best_score:
            best_points = g56_best
            best_score = g56_best_score
    
    return best_points
