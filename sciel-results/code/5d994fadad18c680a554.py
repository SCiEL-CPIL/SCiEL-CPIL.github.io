"""Construct 20 integer points with no four collinear, minimizing largest 3-free subset."""
import time
import random
import math
from itertools import combinations

# Global deadline: must finish well before 150s wall-clock
_DEADLINE = 145.0  # seconds
_start_time = None


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        # Same point; shouldn't happen for distinct points
        return (x1, y1, 0, 0)
    dx //= g
    dy //= g
    # Canonicalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Canonical point on the line: use the one with smaller (x,y) lex
    if (x1, y1) <= (x2, y2):
        cx, cy = x1, y1
    else:
        cx, cy = x2, y2
    return (cx, cy, dx, dy)


def _count_collinear_triples(points):
    """Count number of collinear triples among the points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            # Count points on this line
            on_line = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if points[k] is on the line through i,j
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    on_line += 1
            # If on_line >= 1, then we have a collinear triple (i,j,k)
            # But we're counting per pair, so each triple is counted 3 times
            # Actually, let's just count distinct triples
    # Better: iterate over all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    count += 1
    return count


def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            on_line = 2  # i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    on_line += 1
                    if on_line >= 4:
                        return True
    return False


def _max_3_free_subset(points):
    """
    Find the size of the largest subset of points with no three collinear.
    Uses branch and bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity: coll[i][j][k] is True if i,j,k are collinear
    # Actually, let's precompute for each pair (i,j), the set of k that are collinear with them
    collinear_with_pair = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    collinear_with_pair[i][j].add(k)
                    collinear_with_pair[j][i].add(k)
    
    # Also precompute: for each pair of points already in the subset,
    # which other points would create a collinear triple
    # For a subset S, a point p can be added if for all pairs (a,b) in S,
    # p is not collinear with a and b.
    
    best = [0]
    
    def _search(subset_mask, current_size, candidate_mask):
        if _time_left() < 0.5:
            return
        
        if current_size > best[0]:
            best[0] = current_size
        
        if candidate_mask == 0:
            return
        
        # Upper bound: current_size + popcount(candidate_mask)
        if current_size + bin(candidate_mask).count('1') <= best[0]:
            return
        
        # Try adding each candidate point
        # Order candidates heuristically: try points that are "most collinear" first
        # to find conflicts early
        candidates = [i for i in range(n) if (candidate_mask >> i) & 1]
        
        for idx in candidates:
            new_mask = subset_mask | (1 << idx)
            # Check if adding idx creates any collinear triple
            valid = True
            # For each pair in subset, check if idx is collinear with them
            for i in range(n):
                if not ((subset_mask >> i) & 1):
                    continue
                for j in range(i + 1, n):
                    if not ((subset_mask >> j) & 1):
                        continue
                    if idx in collinear_with_pair[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also need to check: for pairs involving idx and existing points,
            # no other point in subset is collinear
            # Actually the above check covers all pairs in subset.
            # We also need: for any two existing points a,b in subset, idx not collinear with them.
            # That's what we checked.
            
            new_candidate_mask = candidate_mask & ~(1 << idx)
            _search(new_mask, current_size + 1, new_candidate_mask)
            
            if _time_left() < 0.5:
                return
    
    _search(0, 0, (1 << n) - 1)
    return best[0]


def _generate_random_points():
    """Generate 20 random distinct integer points in [-100,100]^2."""
    points = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        points.add((x, y))
    return list(points)


def _local_search(points, time_limit=10.0):
    """
    Local search to minimize the largest 3-free subset size.
    Moves: swap one point with a random new point, or move one point to a random location.
    """
    start = time.time()
    current = list(points)
    current_score = _max_3_free_subset(current)
    best = list(current)
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start < time_limit and _time_left() > 5.0:
        # Generate a neighbor
        move_type = random.random()
        if move_type < 0.5:
            # Move one point to a random location
            new_pts = list(current)
            idx = random.randint(0, 19)
            while True:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) not in [(p[0], p[1]) for p in new_pts if new_pts.index(p) != idx]:
                    break
            new_pts[idx] = (nx, ny)
        else:
            # Swap two points with random new points
            new_pts = list(current)
            i, j = random.sample(range(20), 2)
            used = set((p[0], p[1]) for p in new_pts)
            while True:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) not in used:
                    break
            used.discard((new_pts[i][0], new_pts[i][1]))
            used.add((nx, ny))
            while True:
                nx2 = random.randint(-100, 100)
                ny2 = random.randint(-100, 100)
                if (nx2, ny2) not in used:
                    break
            used.discard((new_pts[j][0], new_pts[j][1]))
            used.add((nx2, ny2))
            new_pts[i] = (nx, ny)
            new_pts[j] = (nx2, ny2)
        
        if _has_four_collinear(new_pts):
            T *= cooling
            continue
        
        new_score = _max_3_free_subset(new_pts)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = list(current)
        
        T *= cooling
    
    return best, best_score


def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Start with a good initial configuration
    # Try several initial configurations
    best_points = None
    best_score = 20  # worst case
    
    # Strategy 1: Start from random and do local search
    for attempt in range(3):
        if _time_left() < 20.0:
            break
        
        init_pts = _generate_random_points()
        if _has_four_collinear(init_pts):
            continue
        
        pts, score = _local_search(init_pts, time_limit=12.0)
        if score < best_score:
            best_score = score
            best_points = pts
    
    # Strategy 2: Try structured initial configurations
    # Points on a conic-like structure but perturbed
    if _time_left() > 15.0:
        # Use points near a parabola but with small perturbations to break collinearity
        base_pts = []
        for x in range(-10, 10):
            y = x * x
            # Small random perturbation
            px = x + random.randint(-2, 2)
            py = y + random.randint(-3, 3)
            base_pts.append((px, py))
        # Remove duplicates
        seen = set()
        unique_pts = []
        for p in base_pts:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        # Fill if needed
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append((x, y))
        
        if not _has_four_collinear(unique_pts):
            pts, score = _local_search(unique_pts, time_limit=12.0)
            if score < best_score:
                best_score = score
                best_points = pts
    
    # Strategy 3: More aggressive local search from best found
    if best_points is not None and _time_left() > 10.0:
        pts, score = _local_search(best_points, time_limit=15.0)
        if score < best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback: parabola (score 20)
        return [[x, x * x] for x in range(-10, 10)]
    
    return [[p[0], p[1]] for p in best_points]
