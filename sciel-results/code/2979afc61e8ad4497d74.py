"""Improved program: construct 20 integer points with no four collinear,
minimizing the largest subset with no three collinear (exact B&B score)."""
import time
import random
from itertools import combinations
from fractions import Fraction

DEADLINE = 115.0  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize direction vector
    if dx < 0:
        dx, dy = -dx, -dy
    elif dx == 0 and dy < 0:
        dy = -dy
    # GCD reduction
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Line: (dy, -dx) is normal; intercept c = dx*y1 - dy*x1 (in reduced form)
    # Actually use: line through p1 with direction (dx,dy)
    # Normal form: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = [i, j]
            else:
                lines[key].append(i)
                lines[key].append(j)
    # Check for any line with 4+ points
    for key, pts in lines.items():
        # pts has duplicates; count unique
        if len(set(pts)) >= 4:
            return False
    return True

def _max_no_three_collinear(points):
    """Exact branch and bound: largest subset with no three collinear."""
    n = len(points)
    # Precompute for each point, which other points it's collinear with in triples
    # For each pair, find all points on the same line
    # Build: for each point i, set of points j such that there exists k with i,j,k collinear
    # Actually: for each point i, we need to know which subsets of other points contain a triple with i
    
    # Better: precompute all lines with their point sets
    lines_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines_map:
                lines_map[key] = set([i, j])
            else:
                lines_map[key].add(i)
                lines_map[key].add(j)
    
    # For each point, list of "conflict lines" - lines containing this point with 3+ points
    # A triple (i,j,k) is collinear if they're all on the same line
    # For B&B: when we add point i to a set S, we must ensure no two points in S are on the same line with i
    # i.e., for any line L containing i, |S ∩ L| <= 1 (since if 2 points in S are on L with i, that's 3 collinear)
    
    # Precompute: for each point i, list of lines through i that have 3+ points
    point_lines = [[] for _ in range(n)]
    for key, pts in lines_map.items():
        if len(pts) >= 3:
            for p in pts:
                point_lines[p].append(frozenset(pts))
    
    # Deduplicate
    for i in range(n):
        point_lines[i] = list(set(point_lines[i]))
    
    best = 0
    best_set = set()
    
    def bnb(idx, current_set):
        nonlocal best, best_set
        if len(current_set) + (n - idx) <= best:
            return
        if idx == n:
            if len(current_set) > best:
                best = len(current_set)
                best_set = set(current_set)
            return
        
        # Try including point idx
        can_include = True
        for line_pts in point_lines[idx]:
            # Count how many points in current_set are on this line
            cnt = sum(1 for p in current_set if p in line_pts)
            if cnt >= 2:
                can_include = False
                break
        if can_include:
            current_set.add(idx)
            bnb(idx + 1, current_set)
            current_set.discard(idx)
        
        # Try excluding point idx
        bnb(idx + 1, current_set)
    
    bnb(0, set())
    return best

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_score = 20  # worst case
    best_points = None
    
    # Generate candidate point sets
    # Strategy: random sampling from [-100, 100]^2, checking constraints
    
    def try_random_set():
        nonlocal best_score, best_points
        while time.time() - start_time < DEADLINE:
            # Generate 20 random points
            pts = []
            seen = set()
            for _ in range(20):
                for _attempt in range(100):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in seen:
                        seen.add((x, y))
                        pts.append((x, y))
                        break
            if len(pts) < 20:
                continue
            
            # Check no four collinear
            if not _no_four_collinear(pts):
                continue
            
            # Compute score
            score = _max_no_three_collinear(pts)
            if score < best_score:
                best_score = score
                best_points = [[x, y] for x, y in pts]
            
            # Early termination if we hit theoretical lower bound
            # For 20 points, g(20) is at most... we just minimize
            if best_score <= 10:
                break
    
    # Also try structured approaches
    # Points on a circle (no three collinear for generic circles, but we need integers)
    # Use points on y = x^2 mod something? No, need actual integers.
    
    # Try a grid-based approach with perturbation
    def try_structured():
        nonlocal best_score, best_points
        if time.time() - start_time > DEADLINE:
            return
        
        # Start with points on parabola, then perturb some
        base = [(x, x * x) for x in range(-10, 10)]
        # Check if valid
        if _no_four_collinear(base):
            score = _max_no_three_collinear(base)
            if score < best_score:
                best_score = score
                best_points = [[x, y] for x, y in base]
        
        # Try to improve by replacing some points
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                return
            pts = list(base)
            # Replace 1-5 random points
            n_replace = random.randint(1, 5)
            for _ in range(n_replace):
                idx = random.randint(0, 19)
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                pts[idx] = (x, y)
            
            # Check distinct
            if len(set(pts)) < 20:
                continue
            # Check bounds
            if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
                continue
            # Check no four collinear
            if not _no_four_collinear(pts):
                continue
            
            score = _max_no_three_collinear(pts)
            if score < best_score:
                best_score = score
                best_points = [[x, y] for x, y in pts]
    
    try_random_set()
    try_structured()
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
