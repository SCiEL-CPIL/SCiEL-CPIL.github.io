"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
import math

DEADLINE = time.time() + 295  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d


def _is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3


def _greedy_randomized(seed=None):
    """Greedy randomized construction."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    
    chosen = []
    chosen_set = set()
    
    for p in order:
        if p in chosen_set:
            continue
        can = True
        for i, j in itertools.combinations(chosen, 2):
            if _is_isosceles(p, i, j):
                can = False
                break
        if can:
            chosen.append(p)
            chosen_set.add(p)
    
    return chosen


def _compute_blockers(S):
    """
    For each point p, compute blockers[p] = set of pairs (a,b) in S
    such that {p,a,b} is isosceles.
    """
    S_set = set(S)
    blockers = {p: [] for p in S}
    # Also need to track for non-S points
    for p in range(NUM_PTS):
        if p not in blockers:
            blockers[p] = []
    
    for i, a in enumerate(S):
        for j, b in enumerate(S):
            if i >= j:
                continue
            # For every p, check if {p, a, b} is isosceles
            for p in range(NUM_PTS):
                if p == a or p == b:
                    continue
                if _is_isosceles(p, a, b):
                    blockers[p].append((a, b))
    
    return blockers


def _is_maximal_after_removal(S, p, blockers):
    """
    Check if S \\ {p} is maximal, given precomputed blockers for S.
    A set is maximal if every point not in it is blocked (forms isosceles with some pair in S).
    After removing p, we need:
    1. p itself must be blocked by S \\ {p} (i.e., there's a pair in S\{p} that with p forms isosceles)
    2. Every q not in S must be blocked by S \\ {p}
    """
    S_set = set(S)
    
    # Check if p is blocked by S \ {p}
    p_blocked = False
    for a, b in blockers[p]:
        if a != p and b != p:
            p_blocked = True
            break
    if not p_blocked:
        return False
    
    # Check if all q not in S are blocked by S \ {p}
    for q in range(NUM_PTS):
        if q in S_set:
            continue
        q_blocked = False
        for a, b in blockers[q]:
            if a != p and b != p:
                q_blocked = True
                break
        if not q_blocked:
            return False
    
    return True


def _prune(S):
    """
    Iteratively remove points that are not essential for maximality.
    A point is essential if removing it makes the set non-maximal.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        blockers = _compute_blockers(S)
        
        # Try to remove each point
        for p in list(S):
            if _is_maximal_after_removal(S, p, blockers):
                S.remove(p)
                improved = True
                break  # Restart loop to recompute blockers
    
    return S


def _local_search_shrink(S, max_iters=10):
    """
    Local search: try removing pairs of points and refilling minimally.
    """
    best = list(S)
    best_len = len(best)
    
    for iteration in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each single point first
        for idx in range(len(best)):
            candidate = [x for i, x in enumerate(best) if i != idx]
            # Check if candidate can be made maximal by adding few points
            # For now, just check if it's still valid (always is) and try to fill
            # Actually, let's just do the prune approach which is faster
            pass
        
        # Try the prune approach on a copy
        pruned = _prune(best)
        if len(pruned) < best_len:
            best = pruned
            best_len = len(best)
            improved = True
        
        if not improved:
            break
    
    return best


def construct():
    """Construct a minimum maximal no-isosceles set."""
    best = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() < DEADLINE - 5:
        S = _greedy_randomized(seed=num_restarts)
        S = _prune(S)
        
        if len(S) < best_len:
            best = S
            best_len = len(S)
        
        num_restarts += 1
    
    if best is None:
        best = _greedy_randomized(seed=0)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
