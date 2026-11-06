"""Rewritten program: grid-based construction with local search to minimize max cap-free subset."""
import time
import random
import math

_START = time.time()
_DEADLINE = 120.0

def _elapsed():
    return time.time() - _START

def _line_key(p1, p2):
    """Return canonical line key for two points (reduced slope, intercept)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return (0, 0, 0)
    if dx == 0:
        # Vertical line
        return (1, 0, p1[0])
    if dy == 0:
        # Horizontal line
        return (0, 1, p1[1])
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize sign: make dx positive, or if dx==0 make dy positive
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = c
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _build_lines(points):
    """Build dict: line_key -> list of point indices on that line."""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = []
            lines[key].append(i)
            lines[key].append(j)
    # Deduplicate
    for key in list(lines.keys()):
        lines[key] = list(set(lines[key]))
    return lines

def _check_no_four_collinear(points):
    """Check that no line has 4 or more points."""
    lines = _build_lines(points)
    for key, pts in lines.items():
        if len(pts) >= 4:
            return False
    return True

def _get_triple_bitmasks(points):
    """Get bitmask for each collinear triple."""
    n = len(points)
    triple_masks = []
    lines = _build_lines(points)
    for key, pts in lines.items():
        if len(pts) == 3:
            mask = 0
            for idx in pts:
                mask |= (1 << idx)
            triple_masks.append(mask)
    return triple_masks

def _max_cap_free(points, triple_masks):
    """
    Find largest subset of points with no three collinear.
    Uses branch and bound with bitmasks.
    Returns the size of the maximum such subset.
    """
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute for each point, the set of points that form a collinear triple with it
    # and another point. Actually, let's use a different approach.
    # For each point i, precompute which other points j,k form a collinear triple with i.
    
    # Better: for each point i, compute the set of points that share a line with i (pairs).
    # If we select i, we cannot select both j and k where {i,j,k} is a collinear triple.
    
    # Let's use a recursive approach with memoization on the available set.
    # But n=20, so 2^20 = 1M states, which might be feasible.
    
    # Actually, let's try a greedy + local search approach for speed.
    # Or a proper branch and bound.
    
    # For n=20, let's try branch and bound with bitmasks.
    # State: (available_mask, current_size)
    # We want to maximize current_size + max additional from available.
    
    # Precompute: for each triple mask, if we include any point in the triple,
    # we can't include both of the other two.
    
    # Let's use a different formulation:
    # A subset S is "cap-free" (no three collinear) iff for every collinear triple T, |S ∩ T| <= 2.
    
    # Branch and bound:
    # Order points. At each step, decide to include or exclude the next point.
    # Prune if current_size + remaining_count <= best_found.
    
    best = [0]
    
    # Sort points by some heuristic to improve pruning
    # Points involved in more triples should be considered first (or last?)
    
    # Let's try: consider points in order, and at each point decide include/exclude.
    # Include is preferred if it doesn't violate constraints.
    
    # For efficiency, let's precompute for each point i, the set of pairs (j,k) 
    # such that {i,j,k} is a collinear triple.
    
    point_pairs = [[] for _ in range(n)]
    for mask in triple_masks:
        # Find the three indices in this mask
        idxs = [i for i in range(n) if (mask >> i) & 1]
        if len(idxs) == 3:
            a, b, c = idxs
            point_pairs[a].append((b, c))
            point_pairs[b].append((a, c))
            point_pairs[c].append((a, b))
    
    # Now branch and bound
    # available: bitmask of points still available
    # chosen: bitmask of points already chosen
    # For a point i to be addable, for each pair (j,k) in point_pairs[i],
    # not both j and k are in chosen.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def can_add(i, chosen_mask):
        for j, k in point_pairs[i]:
            if (chosen_mask >> j) & 1 and (chosen_mask >> k) & 1:
                return False
        return True
    
    def branch(available_mask, chosen_mask, depth):
        if _elapsed() > _DEADLINE - 5:
            return
        # Upper bound: current chosen count + remaining available
        ub = bin(chosen_mask).count('1') + bin(available_mask).count('1')
        if ub <= best[0]:
            return
        
        if available_mask == 0:
            sz = bin(chosen_mask).count('1')
            if sz > best[0]:
                best[0] = sz
            return
        
        # Find the lowest set bit
        i = (available_mask & (-available_mask)).bit_length() - 1
        new_avail = available_mask & ~(1 << i)
        
        # Option 1: include i
        if can_add(i, chosen_mask):
            branch(new_avail, chosen_mask | (1 << i), depth + 1)
        
        # Option 2: exclude i
        branch(new_avail, chosen_mask, depth + 1)
    
    full_mask = (1 << n) - 1
    branch(full_mask, 0, 0)
    
    return best[0]

def _compute_score(points):
    """Compute the score: largest subset with no three collinear."""
    triple_masks = _get_triple_bitmasks(points)
    return _max_cap_free(points, triple_masks)

def _is_valid(points):
    """Check if points are valid: 20 distinct, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return _check_no_four_collinear(points)

def _gen_perturbed_grid():
    """Generate a grid-based initial configuration."""
    random.seed(42)
    # Start with a grid and perturb
    points = []
    for i in range(5):
        for j in range(4):
            x = (i - 2) * 20 + random.randint(-5, 5)
            y = (j - 2) * 20 + random.randint(-5, 5)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            points.append([x, y])
    
    # Check validity, if not, try again
    attempts = 0
    while not _is_valid(points) and attempts < 100:
        points = []
        for i in range(5):
            for j in range(4):
                x = (i - 2) * 20 + random.randint(-8, 8)
                y = (j - 2) * 20 + random.randint(-8, 8)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                points.append([x, y])
        attempts += 1
    
    return points

def _local_search(points, max_iter=200):
    """Local search: try moving points to improve score."""
    best_points = [list(p) for p in points]
    best_score = _compute_score(best_points)
    
    for iteration in range(max_iter):
        if _elapsed() > _DEADLINE - 10:
            break
        
        improved = False
        for _ in range(50):
            if _elapsed() > _DEADLINE - 10:
                break
            
            cand = [list(p) for p in best_points]
            idx = random.randint(0, 19)
            # Move the point
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            new_x = cand[idx][0] + dx
            new_y = cand[idx][1] + dy
            new_x = max(-100, min(100, new_x))
            new_y = max(-100, min(100, new_y))
            
            # Check distinctness
            cand[idx] = [new_x, new_y]
            if not _is_valid(cand):
                continue
            
            score = _compute_score(cand)
            if score < best_score:
                best_points = cand
                best_score = score
                improved = True
                break
        
        if not improved:
            # Try larger moves
            for _ in range(20):
                if _elapsed() > _DEADLINE - 10:
                    break
                cand = [list(p) for p in best_points]
                idx = random.randint(0, 19)
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                cand[idx] = [new_x, new_y]
                if not _is_valid(cand):
                    continue
                score = _compute_score(cand)
                if score < best_score:
                    best_points = cand
                    best_score = score
                    improved = True
                    break
    
    return best_points, best_score

def construct():
    random.seed(123)
    
    # Generate initial configuration
    points = _gen_perturbed_grid()
    
    # If still not valid, use a fallback
    if not _is_valid(points):
        # Fallback: use a known valid configuration
        # Parabola gives score 20, but we want something better
        points = [[i, i * i] for i in range(-10, 10)]
    
    # Local search
    best_points, best_score = _local_search(points, max_iter=100)
    
    # Verify final result
    if not _is_valid(best_points):
        return [[i, i * i] for i in range(-10, 10)]
    
    return best_points
