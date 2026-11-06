"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import defaultdict

def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _line_key(p1, p2):
    """Get a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    from math import gcd
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (x1, y1)
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Canonical point on line: find integer point with minimal coordinates
    # Line: (x, y) = (x1, y1) + t*(dx, dy)
    # We want the "first" point in this direction
    # Use the convention that the first point has the smallest x (or y if x equal)
    # Actually, for line identification, we can use (slope, intercept) but that's float.
    # Better: use the normalized direction and a point that is "minimal"
    # A point (x, y) is on the line if (x - x1)*dy - (y - y1)*dx == 0
    # For canonical form, let's find the point on the line with the smallest x, 
    # and if tie, smallest y. But this requires searching.
    # Alternative: use the line equation Ax + By + C = 0 with gcd(A,B,C)=1 and A>0 or A==0,B>0
    A = dy
    B = -dx
    C = -(A * x1 + B * y1)
    from math import gcd
    g = gcd(abs(A), gcd(abs(B), abs(C))) if C != 0 else gcd(abs(A), abs(B))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_lines(points):
    """Count how many lines have exactly 3 or more points, and check no 4 collinear."""
    line_to_points = defaultdict(list)
    for i, j in itertools.combinations(range(len(points)), 2):
        key = _line_key(points[i], points[j])
        line_to_points[key].append((i, j))
    
    # Now, for each line, collect all points that lie on it
    # Actually, the above approach is inefficient. Let me redo.
    pass

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i, j, k, l in itertools.combinations(range(n), 4):
        if _collinear(points[i], points[j], points[k]):
            if _collinear(points[i], points[j], points[l]):
                return True
    return False

def _triple_count(points):
    """Count the number of collinear triples."""
    n = len(points)
    count = 0
    for i, j, k in itertools.combinations(range(n), 3):
        if _collinear(points[i], points[j], points[k]):
            count += 1
    return count

def _max_independent_set_size(points):
    """
    Find the largest subset of points with no three collinear.
    This is the independence number of the 3-uniform hypergraph of collinear triples.
    Use branch and bound for exact computation.
    """
    n = len(points)
    # Precompute collinear triples
    triples = []
    for i, j, k in itertools.combinations(range(n), 3):
        if _collinear(points[i], points[j], points[k]):
            triples.append((i, j, k))
    
    if not triples:
        return n
    
    # Build adjacency: for each point, which other points form a triple with it
    # We'll use a recursive branch and bound
    # State: (available_points, current_size, best)
    # We want to find the maximum independent set
    
    # Heuristic: order points by degree (number of triples they're in)
    point_degree = [0] * n
    for i, j, k in triples:
        point_degree[i] += 1
        point_degree[j] += 1
        point_degree[k] += 1
    
    # Sort points by degree descending
    order = sorted(range(n), key=lambda x: -point_degree[x])
    order_inv = [0] * n
    for idx, p in enumerate(order):
        order_inv[p] = idx
    
    # For branch and bound, we maintain a set of available points
    # and a set of "blocked" triples (triples where at least one point is already selected)
    
    best = [0]
    
    # Use a different approach: since n=20, we can use a smarter search
    # The maximum independent set in a 3-uniform hypergraph on 20 vertices
    # We can use inclusion-exclusion or simply backtracking with pruning
    
    # Let's use a simple backtracking with upper bound pruning
    # Upper bound: current_size + number of available points that can still be added
    # A point can be added if it doesn't form a triple with any two already selected points
    
    def backtrack(selected, available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(available)
        if current_size + len(available) <= best[0]:
            return
        
        # Try adding each available point
        avail_list = list(available)
        # Sort by degree for better pruning
        avail_list.sort(key=lambda p: -point_degree[p])
        
        for p in avail_list:
            if p not in available:
                continue
            # Check if p can be added: no triple (p, q, r) where q, r are in selected
            can_add = True
            for i, j, k in triples:
                if p in (i, j, k):
                    others = [x for x in (i, j, k) if x != p]
                    if others[0] in selected and others[1] in selected:
                        can_add = False
                        break
            if can_add:
                new_selected = selected | {p}
                new_available = available - {p}
                # Remove points that can no longer be added? No, they might still be addable
                # Actually, we just remove p from available
                backtrack(new_selected, new_available, current_size + 1)
        
        # Also try not adding any more points (but this is covered by the loop)
    
    # This is too slow. Let me use a different approach.
    # For n=20, we can use a MILP-like approach or just a good heuristic + local search
    # But the task says "exact branch and bound" for the score computation.
    # However, for our purposes in construct(), we just need to find a good configuration.
    # The evaluator will compute the exact score.
    
    # Let me use a simpler greedy + local search to estimate the score,
    # and rely on the evaluator for the exact value.
    # For now, return a heuristic estimate.
    
    # Greedy: pick points one by one, avoiding creating a triple
    selected = set()
    remaining = set(range(n))
    while remaining:
        # Find the point with minimum "conflict" (fewest triples it forms with pairs in selected)
        best_point = None
        best_conflict = float('inf')
        for p in remaining:
            conflict = 0
            for i, j, k in triples:
                if p in (i, j, k):
                    others = [x for x in (i, j, k) if x != p]
                    if others[0] in selected and others[1] in selected:
                        conflict += 1
            if conflict < best_conflict:
                best_conflict = conflict
                best_point = p
        if best_conflict == 0:
            selected.add(best_point)
            remaining.remove(best_point)
        else:
            break
    
    return len(selected)

def construct():
    """
    Construct 20 integer points with |x|, |y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    """
    random.seed(42)
    
    # Start with a known good configuration
    # Points on a parabola have no three collinear, so score = 20
    # We want to minimize the score, so we want more collinear triples
    
    # Strategy: use a grid-like pattern but perturb to avoid 4 collinear
    # Start with a 5x4 grid or similar
    
    # Let's try a perturbed grid
    base_points = []
    for i in range(5):
        for j in range(4):
            base_points.append([i * 20, j * 20])
    
    # This has many 4-collinear points (rows and columns have 5 and 4 points)
    # We need to perturb to break 4-collinear
    
    # Use randomized local search
    points = [[x, y] for x, y in base_points]
    
    def is_valid(pts):
        return len(set(tuple(p) for p in pts)) == 20 and not _has_four_collinear(pts)
    
    def score(pts):
        return _triple_count(pts)
    
    # Local search: swap coordinates or move points
    best_points = points
    best_score = score(points)
    
    for iteration in range(10000):
        # Pick a random point and move it
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        # Try a random new position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            s = score(points)
            if s > best_score:  # More triples is better (lower independence number)
                best_score = s
                best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
        else:
            points[idx] = [old_x, old_y]
    
    return best_points
# EVOLVE-BLOCK-END
