"""Rewritten program: randomized greedy with many restarts + local improvement for minimum maximal no-isosceles set."""
import itertools
import random
import time

N = 16

# Precompute all grid points
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs
# dist2[i][j] = squared distance between POINTS[i] and POINTS[j]
_dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        _dist2[i][j] = d
        _dist2[j][i] = d


def _is_isosceles(a, b, c):
    """Check if triangle (a,b,c) is isosceles (including degenerate/equilateral)."""
    d1 = _dist2[a][b]
    d2 = _dist2[a][c]
    d3 = _dist2[b][c]
    return d1 == d2 or d1 == d3 or d2 == d3


def _greedy_maximal(order):
    """Given an ordering of points, greedily build a maximal no-isosceles set.
    A point is added if it doesn't form an isosceles triangle with any two already-chosen points.
    Returns the list of chosen point indices."""
    chosen = []
    # For efficiency, track for each pair of chosen points whether a candidate would be isosceles
    # We'll just check directly
    for p in order:
        ok = True
        for i in range(len(chosen)):
            a = chosen[i]
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                if _is_isosceles(p, a, b):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p)
    return chosen


def _try_remove(chosen):
    """Try to remove a point from the chosen set while maintaining maximality.
    If removing point at index i still leaves a maximal set (i.e., every unchosen point
    still conflicts with some pair in the reduced set), do it."""
    n = len(chosen)
    if n < 2:
        return chosen
    chosen_set = set(chosen)
    # Precompute: for each unchosen point, which pairs in chosen conflict with it
    # For a point p to be "blocked" by the set S, there must exist a,b in S such that (p,a,b) is isosceles
    # After removing one point r, p is still blocked if there exists a,b in S\{r} with (p,a,b) isosceles
    
    # Let's try removing each point and check maximality
    # To check maximality of S\{r}: for every p not in S\{r}, there must be a,b in S\{r} with isosceles(p,a,b)
    # This is expensive, so let's be smart:
    # For each unchosen point p, let conflicts(p) = set of points in chosen that participate in an isosceles triangle with p and another chosen point
    # Actually, for p to be blocked, we need a pair (a,b) in chosen such that isosceles(p,a,b).
    # If we remove r, p is still blocked if there exists a pair (a,b) in chosen\{r} with isosceles(p,a,b).
    
    # Let's compute for each unchosen point p, the number of "witness pairs" in chosen
    # A witness pair for p is a pair (a,b) in chosen such that isosceles(p,a,b)
    # If count >= 2, then removing any single point still leaves at least one witness pair
    # If count == 1, removing the specific point in that pair might unblock p
    
    unchosen = [p for p in range(NUM_POINTS) if p not in chosen_set]
    
    # For each unchosen p, find witness pairs
    # This is O(|unchosen| * |chosen|^2) which is large. Let's optimize.
    
    # Alternative: just try removing each point and do a quick check
    # For each candidate removal index ri:
    #   new_set = chosen without chosen[ri]
    #   Check: for every p in unchosen (and p != chosen[ri]), is p blocked by new_set?
    #   If yes for all, then removal is valid.
    
    # To speed up, for each unchosen p, precompute the set of chosen points that are "essential" for blocking p.
    # Actually, let's just do it naively but with early termination.
    
    for ri in range(n):
        r = chosen[ri]
        new_set = [c for i, c in enumerate(chosen) if i != ri]
        new_set_set = set(new_set)
        
        # Check if new_set is still maximal
        valid = True
        for p in unchosen:
            if p == r:
                # p was not in chosen, so p != r always (r is in chosen)
                pass
            # Check if p is blocked by new_set
            blocked = False
            for i in range(len(new_set)):
                a = new_set[i]
                for j in range(i + 1, len(new_set)):
                    b = new_set[j]
                    if _is_isosceles(p, a, b):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                valid = False
                break
        if valid:
            return new_set
    return chosen


def _local_improve(chosen):
    """Iteratively try to remove points until no more can be removed."""
    improved = True
    while improved:
        improved = False
        new_chosen = _try_remove(chosen)
        if len(new_chosen) < len(chosen):
            chosen = new_chosen
            improved = True
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin: 300s limit, keep 20s buffer
    
    best = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # First, do a row-major greedy as baseline
    row_major_order = list(range(NUM_POINTS))
    baseline = _greedy_maximal(row_major_order)
    best = baseline
    best_len = len(baseline)
    
    # Now do randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        chosen = _greedy_maximal(order)
        cur_len = len(chosen)
        
        if cur_len < best_len:
            # Try to improve via local search
            improved = _local_improve(chosen)
            cur_len = len(improved)
            if cur_len < best_len:
                best = improved
                best_len = cur_len
        
        # Check deadline
        if time.time() > deadline - 5:
            break
    
    return [POINTS[i] for i in best]


if __name__ == "__main__":
    result = construct()
    print(f"Found maximal no-isosceles set of size {len(result)}")
