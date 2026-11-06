import time
import random
import math
from math import gcd

DEADLINE = 110

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(points):
    lines = _build_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _is_valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return _check_no_four_collinear(points)

def _compute_triples(points):
    """For each point i, compute list of (j,k) pairs where i,j,k collinear, j<k."""
    n = len(points)
    triples = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triples[i].append((j, k))
                    triples[j].append((i, k))
                    triples[k].append((i, j))
    return triples

def _max_cap_free(points):
    """Branch and bound for maximum subset with no three collinear."""
    n = len(points)
    if n < 3:
        return n
    triples = _compute_triples(points)
    
    # Precompute for each point, the set of pairs it conflicts with
    # conflict_mask[i] = bitmask of points j such that i and j are in some triple
    # Actually, we need: when adding point i to current set S, check if any pair in S is collinear with i
    # Precompute: for each i, a list of (j,k) where j<k and i,j,k collinear
    # When checking if i can be added to S: for each (j,k) in triples[i], check if j in S and k in S
    
    # Sort by descending number of triples (more constrained first)
    order = sorted(range(n), key=lambda i: len(triples[i]), reverse=True)
    
    best = 0
    deadline = time.time() + DEADLINE - 5
    
    # Use iterative deepening or simple B&B
    # State: (pos in order, current_mask, current_size)
    # We want to find max size subset with no triple
    
    def can_add(i, mask):
        """Check if point i can be added to set represented by mask."""
        for (j, k) in triples[i]:
            if (mask >> j) & 1 and (mask >> k) & 1:
                return False
        return True
    
    def bound(pos, mask, size):
        """Upper bound on final size."""
        remaining = n - pos
        return size + remaining
    
    def solve(pos, mask, size):
        nonlocal best
        if time.time() > deadline:
            return
        if size > best:
            best = size
        if pos == n:
            return
        # Prune: even if we add all remaining, can we beat best?
        if size + (n - pos) <= best:
            return
        
        i = order[pos]
        # Try including i
        if can_add(i, mask):
            solve(pos + 1, mask | (1 << i), size + 1)
        # Try excluding i
        solve(pos + 1, mask, size)
    
    solve(0, 0, 0)
    return best

def _greedy_construct(rng):
    """Greedy construction: build points one by one, preferring positions that create many collinear triples."""
    points = []
    used = set()
    
    # Start with a few anchor points to create structure
    # Use a perturbed grid-like structure
    base_xs = [-40, -20, 0, 20, 40]
    base_ys = [-30, -10, 10, 30]
    
    candidates = []
    for x in base_xs:
        for y in base_ys:
            px = x + rng.randint(-3, 3)
            py = y + rng.randint(-3, 3)
            px = max(-100, min(100, px))
            py = max(-100, min(100, py))
            candidates.append((px, py))
    
    # Shuffle candidates
    rng.shuffle(candidates)
    
    for (x, y) in candidates:
        if len(points) >= 20:
            break
        if (x, y) in used:
            continue
        # Check no four collinear with existing points
        new_points = points + [[x, y]]
        if _check_no_four_collinear(new_points):
            points.append([x, y])
            used.add((x, y))
    
    # Fill remaining with random points
    while len(points) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) in used:
            continue
        new_points = points + [[x, y]]
        if _check_no_four_collinear(new_points):
            points.append([x, y])
            used.add((x, y))
    
    return points[:20]

def _try_fix(points, rng, max_iters=200):
    """Fix any 4-collinear violations by perturbing points."""
    points = [list(p) for p in points]
    for _ in range(max_iters):
        if _check_no_four_collinear(points):
            return points
        
        lines = _build_lines(points)
        bad_line = None
        bad_pts = None
        for key, pts in lines.items():
            if len(pts) >= 4:
                bad_line = key
                bad_pts = list(pts)
                break
        
        if not bad_pts:
            return points
            
        idx_to_move = rng.choice(bad_pts)
        p = points[idx_to_move]
        
        for _ in range(30):
            new_x = p[0] + rng.randint(-5, 5)
            new_y = p[1] + rng.randint(-5, 5)
            new_x = max(-100, min(100, new_x))
            new_y = max(-100, min(100, new_y))
            
            if any(new_x == q[0] and new_y == q[1] for q in points):
                continue
                
            old_p = points[idx_to_move]
            points[idx_to_move] = [new_x, new_y]
            
            if _check_no_four_collinear(points):
                return points
            
            points[idx_to_move] = old_p
            
    return None

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Step 1: Generate a valid initial configuration using greedy construction
    best_points = None
    best_score = 21  # Worst possible is 20
    
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
        rng.seed(seed)
        pts = _greedy_construct(rng)
        if _is_valid(pts):
            # Evaluate score
            score = _max_cap_free(pts)
            if score < best_score:
                best_score = score
                best_points = [list(p) for p in pts]
    
    if best_points is None:
        # Fallback: simple valid configuration
        best_points = _greedy_construct(rng)
        if not _is_valid(best_points):
            # Parabola fallback
            best_points = [[x, x*x] for x in range(-10, 10)]
        best_score = _max_cap_free(best_points)
    
    # Step 2: Simulated annealing to improve the score
    if time.time() - start_time < DEADLINE - 10:
        current_points = [list(p) for p in best_points]
        current_score = best_score
        temp = 5.0
        cooling = 0.995
        
        while time.time() - start_time < DEADLINE - 10 and temp > 0.5:
            # Generate a neighbor by moving one point
            new_points = [list(p) for p in current_points]
            idx = rng.randint(0, 19)
            old_x, old_y = new_points[idx]
            new_x = max(-100, min(100, old_x + rng.randint(-3, 3)))
            new_y = max(-100, min(100, old_y + rng.randint(-3, 3)))
            new_points[idx] = [new_x, new_y]
            
            # Check validity
            if not _is_valid(new_points):
                temp *= cooling
                continue
            
            new_score = _max_cap_free(new_points)
            
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
            else:
                # Accept with probability exp(-(delta)/temp)
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
            
            temp *= cooling
    
    return best_points
