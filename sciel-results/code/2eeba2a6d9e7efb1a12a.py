"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
import math
from collections import defaultdict

DEADLINE = time.time() + 290  # Safety margin before 330s kill

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


def _compute_blockers_bitmask(S):
    """
    For each point p, compute a bitmask of indices in S that form isosceles with p.
    Returns: blockers[p] = bitmask where bit i is set if S[i] forms isosceles with p and some other S[j]
    Actually, we need: for each p, which pairs (a,b) in S block p?
    Better approach: for each p, blockers[p] = set of indices in S such that there exists another index in S
    forming isosceles with p and that index.
    
    Simpler: for each pair (a,b) in S, find all p that form isosceles with a,b.
    Then for each p, blockers[p] = set of pairs.
    
    For efficiency with bitmask: 
    - For each point p, compute iso_with[p][q] = True if p,q and some r... no.
    
    Let's use: conflict[p][q] = True if there exists r such that {p,q,r} is isosceles.
    But we need pairs within S.
    
    Alternative efficient approach:
    For each pair (a,b) in S, compute the set of points P_ab that form isosceles with a,b.
    Then for each p in P_ab, p is blocked by pair (a,b).
    
    To check if p is blocked by S \ {x}: p is blocked if there exists a pair (a,b) in S\{x} such that p in P_ab.
    Equivalently: p is NOT blocked by S\{x} if for all pairs (a,b) in S\{x}, p not in P_ab.
    
    Let's precompute for each p: blocker_pairs[p] = list of pairs (a,b) (indices in S) that block p.
    Then p is blocked by S\{x} iff there exists (a,b) in blocker_pairs[p] with a != x and b != x.
    """
    S_set = set(S)
    n = len(S)
    S_idx = {v: i for i, v in enumerate(S)}
    
    # For each point p (0 to NUM_PTS-1), find which pairs in S block it
    # blocker_pairs[p] = list of (i, j) where i,j are indices in S, i<j, and {p, S[i], S[j]} is isosceles
    blocker_pairs = [set() for _ in range(NUM_PTS)]
    
    for i in range(n):
        for j in range(i + 1, n):
            a = S[i]
            b = S[j]
            # Find all p such that {p, a, b} is isosceles
            # p != a, p != b
            for p in range(NUM_PTS):
                if p == a or p == b:
                    continue
                if _is_isosceles(p, a, b):
                    blocker_pairs[p].add((i, j))
    
    return blocker_pairs, S_idx


def _is_maximal_after_removal_fast(S, p_idx, blocker_pairs, S_idx):
    """
    Check if S \\ {p} is maximal, where p_idx is the index of p in S.
    S is a list, p = S[p_idx].
    """
    p = S[p_idx]
    
    # Check if p is blocked by S \ {p}
    # p is blocked if there's a pair (i,j) in blocker_pairs[p] with i != p_idx and j != p_idx
    p_blocked = False
    for (i, j) in blocker_pairs[p]:
        if i != p_idx and j != p_idx:
            p_blocked = True
            break
    if not p_blocked:
        return False
    
    # Check if all q not in S are blocked by S \ {p}
    S_set = set(S)
    for q in range(NUM_PTS):
        if q in S_set:
            continue
        q_blocked = False
        for (i, j) in blocker_pairs[q]:
            if i != p_idx and j != p_idx:
                q_blocked = True
                break
        if not q_blocked:
            return False
    
    return True


def _prune(S):
    """
    Iteratively remove points that are not essential for maximality.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        blocker_pairs, S_idx = _compute_blockers_bitmask(S)
        
        # Try to remove each point
        for p_idx in range(len(S)):
            if _is_maximal_after_removal_fast(S, p_idx, blocker_pairs, S_idx):
                S.pop(p_idx)
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
