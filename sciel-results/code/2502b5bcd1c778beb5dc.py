import time
import random
import sys
import numpy as np

N = 512

def _build_conflict_masks():
    """
    For each x in [1, N], precompute a bitmask (as int) of all elements y such that
    adding y to a set containing x would create a 4-AP where both x and y are in the set
    and the other two terms are also in the set.
    
    Actually, we need: for a set S, x is "blocked" if there exists a 4-AP {a, a+d, a+2d, a+3d}
    where x is one of the four, and the other three are in S.
    
    We precompute for each x a bitmask of "forbidden companions": if S contains x, then any y
    in the forbidden mask of x that is also in S would complete a 4-AP.
    
    More precisely: x is blocked if there exists a 4-AP containing x where the other 3 elements
    are all in S. We can check this by: for each 4-AP containing x, check if the other 3 are in S.
    
    For speed, we precompute for each x a list of "triples" (the other 3 elements of 4-APs containing x).
    Then x is blocked if any triple is entirely in S.
    
    But even better: we can use a bitset representation and check via bitwise AND.
    
    Let's use a different approach: represent S as a Python int bitmask.
    For each x, precompute a set of bitmasks: each bitmask represents one "witness" 4-AP containing x,
    with the bits for the OTHER 3 elements set. Then x is blocked if (S_mask & witness) == witness for any witness.
    """
    # For each x, find all 4-APs containing x
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1
    # x can be at position 0, 1, 2, or 3
    
    # We'll store for each x a list of witness bitmasks (int)
    witnesses = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            # 4-AP: a, b, c, e
            # For x=a: witness has bits b, c, e
            witnesses[a].append((1 << b) | (1 << c) | (1 << e))
            # For x=b: witness has bits a, c, e
            witnesses[b].append((1 << a) | (1 << c) | (1 << e))
            # For x=c: witness has bits a, b, e
            witnesses[c].append((1 << a) | (1 << b) | (1 << e))
            # For x=e: witness has bits a, b, c
            witnesses[e].append((1 << a) | (1 << b) | (1 << c))
    
    return witnesses

def _can_add_mask(S_mask, x, witnesses):
    """Check if x can be added to the set represented by S_mask (int bitmask)."""
    for w in witnesses[x]:
        if (S_mask & w) == w:
            return False
    return True

def _greedy_mask(order, witnesses):
    """Construct a 4-AP free set using greedy with bitmask representation."""
    S_mask = 0
    for x in order:
        if _can_add_mask(S_mask, x, witnesses):
            S_mask |= (1 << x)
    return S_mask

def _mask_to_list(S_mask):
    """Convert bitmask to sorted list of elements."""
    return [i for i in range(1, N + 1) if (S_mask >> i) & 1]

def _local_search(S_mask, witnesses, deadline):
    """
    Local search: try to improve the set by:
    1. Adding elements that are currently blocked but might become unblocked after removals
    2. Swapping: remove one element, add another
    
    We do a simple hill-climbing: repeatedly try to add any element that can be added.
    Then try swaps: for each element in the set, try removing it and see if we can add more.
    """
    S_mask = S_mask
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add any element
        for x in range(1, N + 1):
            if not (S_mask >> x) & 1:
                if _can_add_mask(S_mask, x, witnesses):
                    S_mask |= (1 << x)
                    changed = True
        # Try swaps: remove one element, then try to add others
        if time.time() > deadline:
            break
        elements = _mask_to_list(S_mask)
        for rem in elements:
            if time.time() > deadline:
                break
            trial_mask = S_mask & ~(1 << rem)
            added = 0
            for x in range(1, N + 1):
                if not (trial_mask >> x) & 1:
                    if _can_add_mask(trial_mask, x, witnesses):
                        trial_mask |= (1 << x)
                        added += 1
            if added > 1:  # net gain
                S_mask = trial_mask
                changed = True
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Precompute witnesses
    witnesses = _build_conflict_masks()
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Deterministic greedy
    S_det = _greedy_mask(range(1, N + 1), witnesses)
    size_det = bin(S_det).count('1')
    if size_det > best_size:
        best_size = size_det
        best_mask = S_det
    
    # Phase 2: Many random greedy restarts
    # Each restart is fast with bitmask. Estimate: ~1ms per restart.
    # We can do ~50000 restarts in 50 seconds.
    max_restarts = 50000
    restarts = 0
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S_rand = _greedy_mask(candidates, witnesses)
        size_rand = bin(S_rand).count('1')
        if size_rand > best_size:
            best_size = size_rand
            best_mask = S_rand
        restarts += 1
    
    # Phase 3: Local search on the best mask
    if time.time() < deadline - 5.0:
        best_mask = _local_search(best_mask, witnesses, deadline - 5.0)
        best_size = bin(best_mask).count('1')
    
    return _mask_to_list(best_mask)

if __name__ == "__main__":
    res = construct()
    print(len(res))
