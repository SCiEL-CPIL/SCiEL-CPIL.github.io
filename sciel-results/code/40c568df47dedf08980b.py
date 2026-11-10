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


# Precompute conflict_pairs: for each point k, list of (a,b) pairs such that {k,a,b} is isosceles
conflict_pairs = [[] for _ in range(NUM_PTS)]
for k in range(NUM_PTS):
    for a in range(NUM_PTS):
        if a == k:
            continue
        for b in range(a + 1, NUM_PTS):
            if b == k:
                continue
            da = dist[k][a]
            db = dist[k][b]
            dab = dist[a][b]
            if da == db or da == dab or db == dab:
                conflict_pairs[k].append((a, b))


def _can_add_mask(k, S_mask):
    """Check if point k can be added to set represented by bitmask S_mask."""
    for a, b in conflict_pairs[k]:
        if (S_mask >> a) & 1 and (S_mask >> b) & 1:
            return False
    return True


def _is_maximal(S_list):
    """Check if set S_list is maximal."""
    S_mask = 0
    for p in S_list:
        S_mask |= (1 << p)
    S_set = set(S_list)
    for p in range(NUM_PTS):
        if p not in S_set:
            if _can_add_mask(p, S_mask):
                return False
    return True


def _greedy_randomized(seed=None):
    """Greedy randomized construction using bitmask for speed."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    
    chosen = []
    chosen_mask = 0
    chosen_set = set()
    
    for p in order:
        if p in chosen_set:
            continue
        if _can_add_mask(p, chosen_mask):
            chosen.append(p)
            chosen_set.add(p)
            chosen_mask |= (1 << p)
    
    return chosen


def _prune(S):
    """
    Iteratively remove points that are not essential for maximality.
    Uses bitmask-based conflict checking for speed.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
        S_set = set(S)
        
        # Try to remove each point
        for p_idx in range(len(S)):
            p = S[p_idx]
            # Check if S \ {p} is still maximal
            S_mask_without = S_mask & ~(1 << p)
            
            # First check if p itself is blocked by S \ {p}
            # p is blocked if there exists a pair (a,b) in S\{p} such that {p,a,b} is isosceles
            p_blocked = False
            for a, b in conflict_pairs[p]:
                if a == p or b == p:
                    continue
                if (S_mask_without >> a) & 1 and (S_mask_without >> b) & 1:
                    p_blocked = True
                    break
            if not p_blocked:
                # p is not blocked, so removing p would make it addable -> not maximal
                continue
            
            # Check if all q not in S are blocked by S \ {p}
            all_blocked = True
            for q in range(NUM_PTS):
                if q in S_set:
                    continue
                q_blocked = False
                for a, b in conflict_pairs[q]:
                    if a == q or b == q:
                        continue
                    if (S_mask_without >> a) & 1 and (S_mask_without >> b) & 1:
                        q_blocked = True
                        break
                if not q_blocked:
                    all_blocked = False
                    break
            
            if all_blocked:
                S.pop(p_idx)
                improved = True
                break  # Restart loop to recompute
    
    return S


def _local_search_swap(S, max_swaps=50):
    """
    Local search with swap operations: try removing one point and adding another.
    This can find smaller maximal sets that pure removal cannot reach.
    """
    S = list(S)
    S_set = set(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    best = list(S)
    best_len = len(best)
    
    for _ in range(max_swaps):
        if time.time() > DEADLINE - 10:
            break
        
        improved = False
        
        # Try all (remove, add) pairs
        for remove_idx in range(len(S)):
            if time.time() > DEADLINE - 10:
                break
            r = S[remove_idx]
            S_mask_without_r = S_mask & ~(1 << r)
            S_set_without_r = S_set - {r}
            
            # Find candidates to add
            candidates = []
            for q in range(NUM_PTS):
                if q in S_set_without_r:
                    continue
                if _can_add_mask(q, S_mask_without_r):
                    candidates.append(q)
            
            for q in candidates:
                # Try the swap: remove r, add q
                new_S_mask = S_mask_without_r | (1 << q)
                new_S_set = S_set_without_r | {q}
                
                # Check if new set is maximal
                is_max = True
                for p in range(NUM_PTS):
                    if p in new_S_set:
                        continue
                    if _can_add_mask(p, new_S_mask):
                        is_max = False
                        break
                
                if is_max:
                    # Accept the swap
                    S[remove_idx] = q
                    S_set.remove(r)
                    S_set.add(q)
                    S_mask = new_S_mask
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return S


def construct():
    """Construct a minimum maximal no-isosceles set."""
    best = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() < DEADLINE - 5:
        S = _greedy_randomized(seed=num_restarts)
        
        # Prune to minimal
        S = _prune(S)
        
        # Local search with swaps
        S = _local_search_swap(S, max_swaps=20)
        
        # Final prune
        S = _prune(S)
        
        if len(S) < best_len:
            best = S
            best_len = len(S)
        
        num_restarts += 1
    
    if best is None:
        best = _greedy_randomized(seed=0)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
