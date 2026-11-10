"""Rewritten program: search for 20 points minimizing max no-three-collinear subset,
while keeping no four collinear."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from fractions import Fraction

_START = time.time()
_DEADLINE = 150 - 10  # safety margin

def _elapsed():
    return time.time() - _START

def _norm(a, b):
    # normalize direction vector (a,b) to a canonical reduced form
    if a == 0 and b == 0:
        return (0, 0)
    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b
    from math import gcd
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
    return (a, b)

def _max_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute for each pair the set of other points collinear with them
    # collinear_with[i][j] = bitmask of points k != i,j collinear with i,j
    # We'll use bitmasks for speed
    collinear = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = points[i][0]-points[j][0], points[i][1]-points[j][1]
            for k in range(n):
                if k == i or k == j:
                    continue
                c, d = points[i][0]-points[k][0], points[i][1]-points[k][1]
                if a*d - b*c == 0:
                    collinear[i][j] |= (1 << k)
                    collinear[j][i] |= (1 << k)
    
    # We want the largest subset S such that no three points in S are collinear.
    # This is equivalent to: for every pair (i,j) in S, no third point k in S is collinear.
    # Branch and bound:
    best = [0]
    
    def _bound(remaining, current_size, forbidden):
        """
        remaining: bitmask of points still available
        current_size: size of current selected set
        forbidden: bitmask of points that cannot be added (would create 3-collinear)
        """
        avail = remaining & ~forbidden
        if avail == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + popcount(avail)
        if current_size + bin(avail).count('1') <= best[0]:
            return
        # Pick a point from avail
        # Heuristic: pick the point that is in the most collinear triples with others in avail
        # For simplicity, pick the first one
        p = avail & (-avail)  # lowest set bit
        pi = p.bit_length() - 1
        avail_rest = avail ^ p
        
        # If we pick pi, we must forbid all points collinear with pi and any other selected point
        # But we don't track which selected points... 
        # Actually, the constraint is: no three collinear in the final set.
        # If we add pi, then for every other selected point j, all points collinear with pi and j must be forbidden.
        # This is complex. Let's use a different formulation.
        
        # Alternative: the problem is a 3-uniform hypergraph independent set.
        # Vertices = points, edges = collinear triples.
        # We want max independent set.
        # Branch on a vertex: include or exclude.
        
        # Let's redo with a cleaner branch and bound.
        pass
    
    # Let's use a different approach: iterate over all subsets? No, 2^20 is too big.
    # Use branch and bound with the following:
    # State: (candidate_mask, selected_mask)
    # For each candidate, check if it can be added without creating a 3-collinear with selected.
    
    # Precompute: for each triple (i,j,k) that is collinear, store it.
    collinear_triples = []
    for i in range(n):
        for j in range(i+1, n):
            mask = collinear[i][j]
            k = mask & (-mask)
            while k:
                ki = k.bit_length() - 1
                if ki > j:
                    collinear_triples.append((i, j, ki))
                k ^= k
    
    # Branch and bound
    best_size = [0]
    
    def _solve(candidates, selected):
        """
        candidates: bitmask of points we can still consider
        selected: bitmask of points already selected
        """
        if candidates == 0:
            sz = bin(selected).count('1')
            if sz > best_size[0]:
                best_size[0] = sz
            return
        # Upper bound
        if bin(selected).count('1') + bin(candidates).count('1') <= best_size[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex in candidates that appears in the most collinear triples
        # with other candidates
        p = candidates & (-candidates)
        pi = p.bit_length() - 1
        rest = candidates ^ p
        
        # Option 1: exclude pi
        _solve(rest, selected)
        
        # Option 2: include pi
        # Check if pi can be added: for every pair (pi, j) where j in selected, 
        # no third point in selected is collinear with pi and j.
        # Also, after adding pi, we must remove from candidates any point that would 
        # form a collinear triple with pi and some selected point.
        new_selected = selected | p
        new_candidates = rest
        
        # For each j in selected, find points collinear with pi and j, remove them from candidates
        j_mask = selected
        j = j_mask & (-j_mask)
        while j:
            ji = j.bit_length() - 1
            # Points collinear with pi and ji
            coll_mask = collinear[pi][ji]
            # Remove those from new_candidates (except pi and ji themselves, which are already handled)
            new_candidates &= ~coll_mask
            j ^= j
        
        # Also, for any two points a, b in new_selected (including pi), 
        # we need to ensure no third point in new_candidates is collinear with them.
        # But this is already handled by the above for pairs involving pi.
        # For pairs not involving pi, they were already checked when those points were added.
        # So we just need to check pairs involving pi.
        
        _solve(new_candidates, new_selected)
    
    _solve((1 << n) - 1, 0)
    return best_size[0]

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            a, b = points[i][0]-points[j][0], points[i][1]-points[j][1]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                c, d = points[i][0]-points[k][0], points[i][1]-points[k][1]
                if a*d - b*c == 0:
                    count += 1
                    if count >= 4:
                        return True
    return False

def _generate_random_points():
    """Generate 20 random distinct integer points with |x|,|y| <= 100."""
    points = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        points.add((x, y))
    return [list(p) for p in points]

def _local_search(points):
    """Try to improve the configuration by swapping points."""
    points = [list(p) for p in points]
    n = len(points)
    
    # Compute current score
    current_score = _max_no_three_collinear(points)
    
    improved = True
    while improved and _elapsed() < _DEADLINE:
        improved = False
        for i in range(n):
            if _elapsed() >= _DEADLINE:
                break
            # Try replacing point i with a random new point
            for _ in range(5):
                if _elapsed() >= _DEADLINE:
                    break
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_point = [new_x, new_y]
                # Check distinctness
                if any(new_point == p for p in points):
                    continue
                # Check no four collinear
                test_points = points[:]
                test_points[i] = new_point
                if _has_four_collinear(test_points):
                    continue
                new_score = _max_no_three_collinear(test_points)
                if new_score < current_score:
                    points = test_points
                    current_score = new_score
                    improved = True
                    break
    return points, current_score

def construct():
    random.seed(42)
    best_points = None
    best_score = 20  # worst case
    
    # Start with some initial configurations
    initial_configs = []
    
    # Config 1: random
    initial_configs.append(_generate_random_points())
    
    # Config 2: points on a circle (approximate)
    import math
    circle_pts = []
    for k in range(20):
        angle = 2 * math.pi * k / 20
        x = round(50 * math.cos(angle))
        y = round(50 * math.sin(angle))
        circle_pts.append([x, y])
    # Ensure distinct
    seen = set()
    unique_circle = []
    for p in circle_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_circle.append(p)
    while len(unique_circle) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_circle.append([x, y])
    initial_configs.append(unique_circle[:20])
    
    # Config 3: grid-like but perturbed
    grid_pts = []
    for i in range(5):
        for j in range(4):
            x = -40 + i * 20 + random.randint(-3, 3)
            y = -30 + j * 20 + random.randint(-3, 3)
            grid_pts.append([x, y])
    initial_configs.append(grid_pts)
    
    for config in initial_configs:
        if _elapsed() >= _DEADLINE:
            break
        if _has_four_collinear(config):
            continue
        score = _max_no_three_collinear(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
    
    # Local search from best
    if best_points is not None:
        best_points, best_score = _local_search(best_points)
    
    # More random restarts
    attempts = 0
    while _elapsed() < _DEADLINE - 5:
        config = _generate_random_points()
        if _has_four_collinear(config):
            continue
        score = _max_no_three_collinear(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
            # Local search
            best_points, best_score = _local_search(best_points)
        attempts += 1
    
    if best_points is None:
        # Fallback: parabola (score 20)
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
# EVOLVE-BLOCK-END
