"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _line_key(p, q):
    """Canonical key for the line through p and q."""
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    g = abs(dy)
    if dx > 0:
        g = _gcd(abs(dx), abs(dy))
    else:
        g = _gcd(abs(dx), abs(dy))
    if g > 1:
        dx //= g
        dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # c = -a*x - b*y where line is a*x + b*y = c
    a = -dy
    b = dx
    c = a * x1 + b * y1
    # normalize sign: make first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    for key, cnt in line_counts.items():
        # Number of pairs on a line with k points is C(k,2)
        # So if cnt > C(3,2) = 3, then k >= 4
        if cnt > 3:
            return False
    return True

def _get_collinear_triples(points):
    """Return set of frozensets of 3 indices that are collinear."""
    n = len(points)
    triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check if collinear
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triples.add(frozenset([i, j, k]))
    return triples

def _max_cap_free(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound with bitmask.
    """
    n = len(points)
    # Precompute for each point, the set of points that form a collinear triple with it
    # Actually, let's use a different approach:
    # For each pair, find all points on the same line.
    
    # Build line membership: for each line, which points are on it
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    # For each line with >= 3 points, any subset of 3 from those points is a bad triple
    # We want the largest subset S such that no line contains 3 or more points from S
    
    # Branch and bound:
    # State: (current_set_mask, next_point_index, current_size)
    # Upper bound: current_size + (n - next_point_index)
    
    best = [0]
    best_set = [0]  # bitmask
    
    # Sort points to help pruning? Maybe sort by degree (number of lines they're on with >=3 pts)
    # For now, use natural order
    
    # Precompute: for each point, the set of points that would create a violation
    # If we add point p to current set, check if any line already has 2 points from current set
    
    def _can_add(mask, p):
        """Check if adding point p to current set (mask) violates no-3-collinear."""
        # For each line containing p, check if current set has >= 2 points on that line
        for key, pts in line_points.items():
            if p in pts:
                # Count how many points in current mask are on this line
                cnt = 0
                for q in pts:
                    if mask & (1 << q):
                        cnt += 1
                        if cnt >= 2:
                            return False
        return True
    
    def _branch(idx, mask, size):
        if idx >= n:
            if size > best[0]:
                best[0] = size
                best_set[0] = mask
            return
        
        # Upper bound
        if size + (n - idx) <= best[0]:
            return
        
        # Option 1: skip point idx
        _branch(idx + 1, mask, size)
        
        # Option 2: include point idx
        if _can_add(mask, idx):
            _branch(idx + 1, mask | (1 << idx), size + 1)
    
    _branch(0, 0, 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    return _max_cap_free(points)

def _initial_parabola():
    """Generate 20 points on a parabola."""
    return [[x, x * x] for x in range(-10, 10)]

def _random_points():
    """Generate 20 random points."""
    points = []
    used = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in used:
            used.add((x, y))
            points.append([x, y])
    return points

def _swap_search(points, max_iters=10000):
    """
    Targeted 2-swap local search to minimize the max 3-collinear-free subset.
    """
    current = [p[:] for p in points]
    current_score = _score(current)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        # Try swapping two points
        for i in range(20):
            if time.time() > _DEADLINE:
                return current, current_score
            for j in range(i + 1, 20):
                # Swap points i and j with random new points
                old_i = current[i][:]
                old_j = current[j][:]
                
                # Generate random replacements
                new_i = [random.randint(-100, 100), random.randint(-100, 100)]
                new_j = [random.randint(-100, 100), random.randint(-100, 100)]
                
                # Ensure distinctness
                if new_i == new_j:
                    continue
                if any(new_i == current[k] for k in range(20)):
                    continue
                if any(new_j == current[k] for k in range(20)):
                    continue
                if new_i == old_j or new_j == old_i:
                    continue
                
                current[i] = new_i
                current[j] = new_j
                
                if not _check_no_four_collinear(current):
                    current[i] = old_i
                    current[j] = old_j
                    continue
                
                new_score = _score(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_i
                    current[j] = old_j
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try single point moves
            for i in range(20):
                if time.time() > _DEADLINE:
                    return current, current_score
                old_i = current[i][:]
                new_i = [random.randint(-100, 100), random.randint(-100, 100)]
                if any(new_i == current[k] for k in range(20)):
                    continue
                current[i] = new_i
                if not _check_no_four_collinear(current):
                    current[i] = old_i
                    continue
                new_score = _score(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_i
            
            if not improved:
                break
    
    return current, current_score

def construct():
    # Start with parabola
    best_points = _initial_parabola()
    best_score = _score(best_points)
    
    # Try multiple random restarts with local search
    random.seed(42)
    
    while time.time() < _DEADLINE:
        points = _random_points()
        if not _check_no_four_collinear(points):
            continue
        
        improved_points, improved_score = _swap_search(points, max_iters=5000)
        
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_points
        
        if time.time() > _DEADLINE:
            break
    
    return best_points
# EVOLVE-BLOCK-END
