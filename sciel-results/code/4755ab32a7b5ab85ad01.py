import random
import time
import itertools

N = 16
DEADLINE = time.time() + 290  # Safety margin before 300s kill

# Pre-compute isosceles partners
# iso_pairs[p][r] is a set of points q such that (p, r, q) forms an isosceles triangle.
# p, r, q are distinct.
# Note: The triangle is isosceles if d(p,r)==d(p,q) or d(p,r)==d(r,q) or d(p,q)==d(r,q).
# We need to be careful with duplicates and self-loops.

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    # Distinct points assumed
    return d1 == d2 or d1 == d3 or d2 == d3

# Pre-computation
# There are 256 points.
# For each pair (p, r), find all q such that (p, r, q) is isosceles.
# This is O(256^3) which is 16M operations. Acceptable once.
# But we can be smarter.
# For fixed p, r:
# 1. d(pr) is fixed.
# 2. q must satisfy d(p,q) == d(pr) OR d(r,q) == d(pr) OR d(p,q) == d(r,q).
#    Case 1: d(p,q) == d(pr). q is on circle centered p with radius d(pr).
#    Case 2: d(r,q) == d(pr). q is on circle centered r with radius d(pr).
#    Case 3: d(p,q) == d(r,q). q is on perpendicular bisector of pr.
#    We just need to check all 256 points q for each pair (p,r).
#    256 * 256 * 256 = 16,777,216 checks.
#    Each check is 3 distance calcs.
#    In Python, this might take ~1-2 seconds.

_all_points = list(itertools.product(range(N), repeat=2))
_point_set = set(_all_points)
_point_index = {p: i for i, p in enumerate(_all_points)}

# We will store iso_pairs as a list of lists of sets.
# iso_pairs[i][j] = set of indices k such that (i, j, k) is isosceles.
# i != j != k != i.

def _precompute_iso_pairs():
    pairs = [[set() for _ in range(len(_all_points))] for _ in range(len(_all_points))]
    for i in range(len(_all_points)):
        p = _all_points[i]
        for j in range(len(_all_points)):
            if i == j:
                continue
            r = _all_points[j]
            d_pr = _dist2(p, r)
            for k in range(len(_all_points)):
                if k == i or k == j:
                    continue
                q = _all_points[k]
                # Check if (p, r, q) is isosceles
                d_pq = _dist2(p, q)
                d_rq = _dist2(r, q)
                if d_pr == d_pq or d_pr == d_rq or d_pq == d_rq:
                    pairs[i][j].add(k)
    return pairs

_iso_pairs = _precompute_iso_pairs()

def _weighted_greedy(seed_val):
    rng = random.Random(seed_val)
    S = []
    valid_candidates = set(range(len(_all_points)))
    
    while valid_candidates:
        best_p = None
        best_score = -1
        ties = []
        
        for p_idx in valid_candidates:
            # Calculate score: number of points in valid_candidates that would be blocked
            # if we add p_idx.
            # A point q is blocked if it forms an isosceles triangle with p_idx and any r in S.
            blocked = set()
            for r_idx in S:
                # Get points q such that (p_idx, r_idx, q) is isosceles
                blocked.update(_iso_pairs[p_idx][r_idx])
            
            # Count how many of these are in valid_candidates
            score = len(blocked & valid_candidates)
            
            if score > best_score:
                best_score = score
                ties = [p_idx]
            elif score == best_score:
                ties.append(p_idx)
        
        # Pick a point from ties randomly
        p_idx = rng.choice(ties)
        S.append(p_idx)
        
        # Update valid_candidates
        valid_candidates.remove(p_idx)
        
        # Remove points blocked by p_idx relative to existing S
        to_remove = set()
        for r_idx in S:
            to_remove.update(_iso_pairs[p_idx][r_idx])
        
        valid_candidates.difference_update(to_remove)
        
    # Convert indices back to points
    return [_all_points[i] for i in S]

def _is_valid(chosen):
    """Check if chosen is isosceles-free."""
    for a, b, c in itertools.combinations(chosen, 3):
        if _is_isosceles(a, b, c):
            return False
    return True

def _is_maximal(chosen):
    """Check if no grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in _all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _can_add(point, chosen):
    """Check if point can be added to chosen without creating isosceles triangle."""
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(point, a, b):
            return False
    return True

def construct():
    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    attempt = 0
    
    # Run multiple weighted greedy restarts
    while time.time() < DEADLINE:
        attempt += 1
        candidate = _weighted_greedy(attempt)
        
        size = len(candidate)
        if size >= best_size:
            continue
        
        # Verify validity and maximality
        if _is_valid(candidate) and _is_maximal(candidate):
            best_set = candidate
            best_size = size
            # print(f"Found better: size={size} at attempt={attempt}")
            
        if best_size <= 12: # Theoretical lower bound is likely higher, but 12 is a good stop
            break

    if best_set is None:
        # Fallback to deterministic greedy if something went wrong
        baseline = []
        for p in _all_points:
            if _can_add(p, baseline):
                baseline.append(p)
        best_set = baseline
        best_size = len(baseline)

    return best_set
