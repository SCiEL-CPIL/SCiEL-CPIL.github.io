"""Optimized 4-AP-free set construction via randomized greedy + local search."""
import time
import random
import sys

N = 512

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) within [1, 512]
# Each progression is a tuple of 4 indices (0-based or 1-based? We'll use 1-based for clarity)
# Actually, let's use 0-based internally for array indexing, but the set is 1..512.
# Let's use 1-based indices for the elements themselves.
# An element x is in [1, 512].
# A 4-AP is (a, a+d, a+2d, a+3d) where all are in [1, 512].

def precompute_aps():
    """Return list of tuples (a, b, c, d) representing 4-APs in [1, 512]."""
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a, b, c, e))
    return aps

ALL_APS = precompute_aps()

# For each element, precompute which APs it participates in
# element_aps[x] = list of indices of APs that contain x
ELEMENT_APS = [[] for _ in range(N + 1)]
for i, ap in enumerate(ALL_APS):
    for x in ap:
        ELEMENT_APS[x].append(i)

def is_valid_set(selected):
    """Check if selected (a set of ints) contains no 4-AP."""
    for ap in ALL_APS:
        if all(x in selected for x in ap):
            return False
    return True

def greedy_construct(seed=42):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    selected = set()
    # For each candidate, check if adding it creates a 4-AP
    # We need to be careful: adding x is safe iff for every AP containing x,
    # the other 3 elements are not all in selected.
    for x in candidates:
        safe = True
        for ap_idx in ELEMENT_APS[x]:
            ap = ALL_APS[ap_idx]
            others = [y for y in ap if y != x]
            if all(y in selected for y in others):
                safe = False
                break
        if safe:
            selected.add(x)
    return selected

def local_search(selected, max_iters=5000, seed=42):
    """Local search: try to add elements, or swap out one for two."""
    rng = random.Random(seed)
    selected = set(selected)
    best_size = len(selected)
    
    # Build a conflict graph or just use direct checks
    # To speed up, we can track for each AP how many of its elements are in selected.
    # ap_count[i] = number of elements of AP i that are in selected.
    # If ap_count[i] == 4, it's a violation.
    
    # Let's use a more efficient local search with swap moves.
    # Move types:
    # 1. Add an element x not in selected (if it doesn't create a 4-AP)
    # 2. Remove an element y from selected and add two elements x1, x2 not in selected
    #    such that the new set is valid and size increases by 1.
    # 3. Swap: remove y, add x (size stays same, but might enable future adds)
    
    # For simplicity and speed, let's do:
    # - Repeatedly try to add any element that can be added safely.
    # - If no element can be added, try random swaps: remove a random y, 
    #   then try to add two random elements x1, x2.
    
    # Actually, a simpler and effective heuristic:
    # 1. Start with a greedy set.
    # 2. Try to add elements in random order until stuck.
    # 3. If stuck, pick a random element y to remove, then try to add elements again.
    
    current = set(selected)
    # Sort elements not in current by some heuristic (e.g., degree in conflict graph)
    
    for _ in range(max_iters):
        # Try to add elements
        added_any = False
        # Random order of candidates
        candidates = [x for x in range(1, N + 1) if x not in current]
        rng.shuffle(candidates)
        for x in candidates:
            # Check if x can be added
            safe = True
            for ap_idx in ELEMENT_APS[x]:
                ap = ALL_APS[ap_idx]
                others = [y for y in ap if y != x]
                if all(y in current for y in others):
                    safe = False
                    break
            if safe:
                current.add(x)
                added_any = True
        
        if not added_any:
            # Stuck. Try a random swap: remove a random element, then try to add.
            if len(current) == 0:
                break
            y = rng.choice(list(current))
            current.remove(y)
            # Now try to add elements again
            candidates = [x for x in range(1, N + 1) if x not in current]
            rng.shuffle(candidates)
            for x in candidates:
                safe = True
                for ap_idx in ELEMENT_APS[x]:
                    ap = ALL_APS[ap_idx]
                    others = [z for z in ap if z != x]
                    if all(z in current for z in others):
                        safe = False
                        break
                if safe:
                    current.add(x)
                    # Continue adding
                    # We need to re-check other candidates, but for simplicity, 
                    # we just do one pass. A more thorough version would loop.
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Try multiple random seeds for greedy + local search
    seed = 0
    while time.time() < deadline:
        if seed == 0:
            # First, try a deterministic greedy for a baseline
            s = greedy_construct(seed=0)
            s = local_search(s, max_iters=2000, seed=0)
        else:
            s = greedy_construct(seed=seed)
            s = local_search(s, max_iters=2000, seed=seed)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        
        seed += 1
    
    # Final check
    if not best_set:
        # Fallback: simple greedy
        best_set = greedy_construct(seed=123)
    
    # Verify validity (just in case)
    if not is_valid_set(best_set):
        # This should not happen, but if it does, do a cleanup
        # Remove elements involved in 4-APs
        changed = True
        while changed:
            changed = False
            for ap in ALL_APS:
                if all(x in best_set for x in ap):
                    best_set.remove(ap[0])
                    changed = True
    
    return sorted(best_set)
