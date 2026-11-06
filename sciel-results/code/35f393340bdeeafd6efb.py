"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _line_through_two(p1, p2):
    """Return normalized line (dx, dy, c) for line through p1, p2.
    Line: dx*(x - p1x) - dy*(y - p1y) = 0  =>  dx*x - dy*y = c
    where dx, dy are normalized so gcd(|dx|,|dy|)=1 and dx>=0 (or dx==0,dy>0).
    c = dx*p1x - dy*p1y
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Normalize sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * p1[0] - dy * p1[1]
    return (dx, dy, c)

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map from line key to set of point indices on that line
    line_to_pts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_through_two(points[i], points[j])
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    
    count = 0
    for key, pts in line_to_pts.items():
        if len(pts) == 4:
            count += 1
    return count

def _no_five_collinear(points):
    """Check that no 5 points are collinear."""
    n = len(points)
    line_to_pts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_through_two(points[i], points[j])
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    for key, pts in line_to_pts.items():
        if len(pts) >= 5:
            return False
    return True

def _score(points):
    """Return score: number of 4-point lines if valid, else -1."""
    if len(points) != 16:
        return -1
    if len(set(tuple(p) for p in points)) != 16:
        return -1
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1
    if not _no_five_collinear(points):
        return -1
    return _count_lines_exactly_4(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    if best_score < 0:
        return best_points
    
    # Local search: try replacing one point at a time
    rng = random.Random(42)
    
    # Generate candidate replacement points
    # Focus on points that might create more 4-point lines
    candidates = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            candidates.append([x, y])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates for diversity
        rng.shuffle(candidates)
        
        for idx in range(16):
            if time.time() >= deadline:
                break
            orig_point = best_points[idx]
            for cand in candidates:
                if cand == orig_point:
                    continue
                # Check if cand is already in the set
                if any(cand == p for p in best_points):
                    continue
                # Try replacing
                new_points = best_points[:]
                new_points[idx] = cand
                s = _score(new_points)
                if s > best_score:
                    best_points = new_points
                    best_score = s
                    improved = True
                    break
    
    return best_points
# EVOLVE-BLOCK-END
