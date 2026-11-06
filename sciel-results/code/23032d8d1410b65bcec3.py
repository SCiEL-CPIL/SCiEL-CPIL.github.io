import time
import random
import itertools
from math import gcd

DEADLINE = 145  # safety margin before 150s kill

def _line_key(p1, p2):
    """Return canonical key for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal form: line is dx*(y-y1) - dy*(x-x1) = 0
    # Key: (dy, dx, dy*x1 - dx*y1) normalized
    # Use: a*x + b*y = c where a=-dy, b=dx, c=dx*y1 - dy*x1
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    # Normalize sign: make (a,b) have a>0 or a==0,b>0
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _get_collinear_triples(points):
    """Get all collinear triples (i,j,k) where i<j<k."""
    n = len(points)
    triples = []
    # Group points by line
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append((i, j))
    
    # For each line, find all triples
    for key, pairs in line_map.items():
        # Collect all points on this line
        pts_on_line = set()
        for i, j in pairs:
            pts_on_line.add(i)
            pts_on_line.add(j)
        pts_list = sorted(pts_on_line)
        if len(pts_list) >= 3:
            for t in itertools.combinations(pts_list, 3):
                triples.append(t)
    return triples

def _check_no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return False
    return True

def _max_cap_free(points, triples_set, time_limit=5.0):
    """Find max subset with no three collinear using branch and bound."""
    n = len(points)
    start_time = time.time()
    
    # triples_set is a set of frozensets or tuples of 3 indices
    # For each pair, which third points form a triple?
    # triple_conflict[i][j] = set of k such that {i,j,k} is a triple
    triple_conflict = [set() for _ in range(n)]
    for (i, j, k) in triples_set:
        triple_conflict[i].add((j, k))
        triple_conflict[j].add((i, k))
        triple_conflict[k].add((i, j))
    
    # For branch and bound: we want max subset S such that no triple is fully in S
    # Equivalent: for each triple {i,j,k}, not all three in S
    
    # Use bitmasks for efficiency
    # conflict_mask[i] = bitmask of j such that... actually let's use a different approach
    
    # For each point i, define forbidden[i] as a bitmask of pairs that can't both be with i
    # Actually, let's use: for subset S, it's valid if for every triple {a,b,c}, S doesn't contain all three
    
    # Branch and bound:
    # State: current set S (bitmask), next candidate index
    # Pruning: upper bound = |S| + (n - next_idx)
    
    best = [0]
    best_set = [0]
    
    def bound(current_mask, next_idx):
        """Upper bound on size achievable."""
        return current_mask.bit_count() + (n - next_idx)
    
    def is_valid_add(current_mask, idx):
        """Check if adding idx to current_mask keeps no triple fully contained."""
        # For each triple containing idx, check if the other two are in current_mask
        for other1, other2 in triple_conflict[idx]:
            if (current_mask >> other1) & 1 and (current_mask >> other2) & 1:
                return False
        return True
    
    def dfs(current_mask, next_idx):
        if time.time() - start_time > time_limit:
            return
        cur_size = current_mask.bit_count()
        if cur_size > best[0]:
            best[0] = cur_size
            best_set[0] = current_mask
        # Upper bound
        if bound(current_mask, next_idx) <= best[0]:
            return
        for idx in range(next_idx, n):
            if time.time() - start_time > time_limit:
                return
            # Try including idx
            if is_valid_add(current_mask, idx):
                dfs(current_mask | (1 << idx), idx + 1)
            # Try excluding idx
            dfs(current_mask, idx + 1)
    
    dfs(0, 0)
    return best[0]

def _score(points):
    """Compute the max cap-free subset size."""
    if not _check_no_four_collinear(points):
        return 999  # invalid
    triples = _get_collinear_triples(points)
    triples_set = set(triples)
    return _max_cap_free(points, triples_set, time_limit=8.0)

def _generate_grid_perturbed(seed):
    """Generate a perturbed 5x4 grid."""
    rng = random.Random(seed)
    base_x = list(range(5))
    base_y = list(range(4))
    points = []
    for i in range(5):
        for j in range(4):
            x = base_x[i] + rng.randint(-2, 2)
            y = base_y[j] * 2 + rng.randint(-1, 1)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            points.append([x, y])
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _generate_random(seed):
    """Generate random points."""
    rng = random.Random(seed)
    points = []
    for _ in range(20):
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        points.append([x, y])
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _local_search(points, time_limit=10.0):
    """Local search: try moving points to reduce score."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_score = _score(current)
    if current_score >= 999:
        return current, current_score
    
    best = current
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            ox, oy = current[i]
            # Try small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = ox + dx
                    ny = oy + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    current[i] = [nx, ny]
                    # Check distinctness
                    if len(set(tuple(p) for p in current)) != 20:
                        continue
                    s = _score(current)
                    if s < best_score:
                        best_score = s
                        best = [p[:] for p in current]
                        improved = True
                    current[i] = [ox, oy]
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    
    # Try multiple strategies
    best_points = None
    best_score = 999
    
    # Strategy 1: Perturbed grids
    for seed in range(50):
        if time.time() - start_time > DEADLINE - 20:
            break
        pts = _generate_grid_perturbed(seed)
        if pts is None:
            continue
        if not _check_no_four_collinear(pts):
            continue
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            # Try local search if promising
            if s <= 18 and time.time() - start_time < DEADLINE - 30:
                refined, refined_score = _local_search(pts, time_limit=min(8.0, DEADLINE - 25 - (time.time() - start_time)))
                if refined_score < best_score:
                    best_score = refined_score
                    best_points = refined
    
    # Strategy 2: Random points
    for seed in range(30):
        if time.time() - start_time > DEADLINE - 15:
            break
        pts = _generate_random(seed)
        if pts is None:
            continue
        if not _check_no_four_collinear(pts):
            continue
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    # Fallback: parabola (always valid, score 20)
    if best_points is None or best_score >= 20:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
