"""Rewritten program: constructive line-packing + local search to minimize the largest 3-free subset."""
import time
import random
from itertools import combinations
from fractions import Fraction

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 115  # seconds, safety margin before 120s

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Normalize direction: ensure first non-zero component is positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_collinear_triples(points):
    """Count the number of collinear triples among the given points."""
    n = len(points)
    count = 0
    # Group points by line
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
    for key, idx_set in line_points.items():
        k = len(idx_set)
        if k >= 3:
            count += k * (k - 1) * (k - 2) // 6
    return count

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
    for key, idx_set in line_points.items():
        if len(idx_set) >= 4:
            return True
    return False

def _max_3free_subset(points):
    """Find the largest subset with no three collinear using branch and bound."""
    n = len(points)
    # Precompute for each pair, the set of points collinear with both
    collinear_with = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            for k in range(n):
                if k != i and k != j:
                    key2 = _line_key(points[i], points[k])
                    if key2 == key:
                        collinear_with[i].add(k)
                        collinear_with[j].add(k)
    
    # Build conflict graph: edge (i,j) means they can't both be in a 3-free set if there's a k collinear with both
    # Actually, for 3-free: a set S is 3-free if no three points in S are collinear.
    # This is equivalent to: for every line, S contains at most 2 points from that line.
    
    # Use branch and bound with greedy coloring
    best = [0]
    
    def _branch(selected, remaining, depth):
        if _time_remaining() < 0.5:
            return
        if len(remaining) == 0:
            if len(selected) > best[0]:
                best[0] = len(selected)
            return
        
        # Upper bound: current selected + all remaining
        ub = len(selected) + len(remaining)
        if ub <= best[0]:
            return
        
        # Greedy lower bound: try to greedily add from remaining
        greedy = list(selected)
        rem = list(remaining)
        # For each point in rem, check if adding it keeps 3-free property
        # A point p can be added if for every other point q in greedy, 
        # the number of points in greedy collinear with p and q is < 2
        # Simpler: maintain line counts
        line_counts = {}
        for i in range(len(greedy)):
            for j in range(i + 1, len(greedy)):
                key = _line_key(points[greedy[i]], points[greedy[j]])
                if key is not None:
                    line_counts[key] = line_counts.get(key, 0) + 1
        
        for p in rem:
            can_add = True
            for q in greedy:
                key = _line_key(points[p], points[q])
                if key is not None:
                    if line_counts.get(key, 0) >= 2:
                        can_add = False
                        break
            if can_add:
                greedy.append(p)
                # Update line counts
                for q in greedy[:-1]:
                    key = _line_key(points[p], points[q])
                    if key is not None:
                        line_counts[key] = line_counts.get(key, 0) + 1
                if len(greedy) > best[0]:
                    best[0] = len(greedy)
        
        # Branch: pick a point to branch on
        # Heuristic: pick the point with most conflicts
        pivot = rem[0]
        # Try including pivot
        new_selected = selected + [pivot]
        # Filter remaining: remove points that would create a triple with two in new_selected
        new_remaining = []
        for r in rem[1:]:
            # Check if r is collinear with any pair in new_selected
            conflict = False
            for a_idx in range(len(new_selected)):
                for b_idx in range(a_idx + 1, len(new_selected)):
                    a = new_selected[a_idx]
                    b = new_selected[b_idx]
                    key_ab = _line_key(points[a], points[b])
                    key_ar = _line_key(points[a], points[r])
                    if key_ab is not None and key_ab == key_ar:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                new_remaining.append(r)
        
        _branch(new_selected, new_remaining, depth + 1)
        
        # Try excluding pivot
        _branch(selected, rem[1:], depth + 1)
    
    _branch([], list(range(n)), 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    if _has_four_collinear(points):
        return None  # Invalid
    return _max_3free_subset(points)

def _construct_initial():
    """Construct an initial configuration with deliberate collinearities."""
    points = []
    # Place points on a few lines to create many collinear triples
    # Line 1: y = 0, points at x = -10, -5, 0, 5 (4 points - wait, no four collinear!)
    # We need at most 3 per line.
    # Let's place 3 points on several lines.
    
    # Line y=0: (-10,0), (-5,0), (0,0)
    # Line y=1: (5,1), (10,1), (15,1) - but |x|<=100, ok
    # Line y=2: (-8,2), (-3,2), (2,2)
    # Line y=3: (7,3), (12,3), (17,3)
    # Line y=-1: (-15,-1), (-10,-1), (-5,-1)
    # Line y=-2: (0,-2), (5,-2), (10,-2)
    # That's 18 points. Need 2 more.
    # Add 2 more points not creating 4-collinear.
    
    pts = [
        [-10, 0], [-5, 0], [0, 0],
        [5, 1], [10, 1], [15, 1],
        [-8, 2], [-3, 2], [2, 2],
        [7, 3], [12, 3], [17, 3],
        [-15, -1], [-10, -1], [-5, -1],
        [0, -2], [5, -2], [10, -2],
    ]
    # Add 2 more
    pts.append([-20, 5])
    pts.append([20, 5])
    
    # Check no four collinear
    if _has_four_collinear(pts):
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return pts

def _local_search(points, max_iters=5000):
    """Local search: try to improve by moving points."""
    current_score = _score(points)
    if current_score is None:
        return points
    
    best_points = points[:]
    best_score = current_score
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Try moving a random point to a random nearby position
        idx = random.randint(0, len(points) - 1)
        x, y = points[idx]
        new_x = x + random.randint(-3, 3)
        new_y = y + random.randint(-3, 3)
        
        if abs(new_x) > 100 or abs(new_y) > 100:
            continue
        
        new_point = [new_x, new_y]
        # Check distinctness
        if any(p == new_point for i, p in enumerate(points) if i != idx):
            continue
        
        points[idx] = new_point
        
        # Check no four collinear
        if _has_four_collinear(points):
            points[idx] = [x, y]
            continue
        
        new_score = _score(points)
        if new_score is None:
            points[idx] = [x, y]
            continue
        
        if new_score < best_score:
            best_score = new_score
            best_points = points[:]
        else:
            points[idx] = [x, y]
    
    return best_points

def construct():
    # Initial construction
    points = _construct_initial()
    
    # Local search
    points = _local_search(points, max_iters=3000)
    
    # If time remains, try more iterations
    if _time_remaining() > 10:
        points = _local_search(points, max_iters=3000)
    
    # Final validation
    if len(points) != 20:
        return [[x, x * x] for x in range(-10, 10)]
    
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return [[x, x * x] for x in range(-10, 10)]
    
    if len(set(tuple(p) for p in points)) != 20:
        return [[x, x * x] for x in range(-10, 10)]
    
    if _has_four_collinear(points):
        return [[x, x * x] for x in range(-10, 10)]
    
    return points
