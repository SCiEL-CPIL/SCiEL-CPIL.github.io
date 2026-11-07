import time
import random
import itertools

N = 128
DEADLINE = 55.0

# Precompute forbidden sets
# A forbidden set is a set of values {a, b, c, a+b, a+c, b+c} 
# where a,b,c are distinct.
# If all values in this set are in S, it's a violation.
_FORBIDDEN_SETS = []
_FORBIDDEN_SETS_SET = set()
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 > N:
                continue
            # The set of values involved
            vals = {a, b, c, s1, s2, s3}
            # We only care if all are in S.
            # Store the frozenset to avoid duplicates
            fs = frozenset(vals)
            if fs not in _FORBIDDEN_SETS_SET:
                _FORBIDDEN_SETS_SET.add(fs)
                _FORBIDDEN_SETS.append(fs)

# Index: for each value v, which forbidden sets contain it?
_val_to_fsets = {v: [] for v in range(1, N + 1)}
for idx, fs in enumerate(_FORBIDDEN_SETS):
    for v in fs:
        _val_to_fsets[v].append(idx)

def _is_valid_set(sel_set):
    """Check if sel_set (a set of ints) is valid."""
    # A set is invalid if it contains any forbidden set.
    # Efficient check: for each forbidden set, check if it's a subset.
    # Since we build incrementally, we can check on add.
    # But for initial validation of a base set:
    for fs in _FORBIDDEN_SETS:
        if fs.issubset(sel_set):
            return False
    return True

def _greedy_fill_gap(base_set, gap_candidates, deadline):
    """
    Try to add elements from gap_candidates to base_set.
    Returns the new set.
    """
    sel = set(base_set)
    # counts[fs_idx] = number of elements of fs in sel
    counts = [0] * len(_FORBIDDEN_SETS)
    for v in sel:
        for idx in _val_to_fsets[v]:
            counts[idx] += 1
    
    # Sort candidates: maybe by degree (how many fsets they touch) to be safe?
    # Or random.
    candidates = list(gap_candidates)
    random.shuffle(candidates)
    
    added = []
    for v in candidates:
        if time.time() > deadline:
            break
        if v in sel:
            continue
        # Check if adding v completes any forbidden set
        ok = True
        for idx in _val_to_fsets[v]:
            if counts[idx] == len(_FORBIDDEN_SETS[idx]) - 1:
                # Adding v makes it complete
                ok = False
                break
        if ok:
            sel.add(v)
            added.append(v)
            for idx in _val_to_fsets[v]:
                counts[idx] += 1
    return sel

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best_set = set(range(65, 129))  # Seed fallback
    best_size = len(best_set)
    
    # Strategy: Iterate over possible A and B blocks
    # B is [L_B, 128]
    # A is [L_A, R_A] with R_A < L_B
    # Gap is (R_A, L_B)
    
    # We know A=[16,32], B=[64,128] is good.
    # Let's search around it.
    
    # Precompute valid A blocks that don't have internal violations
    # An A block is valid if no a,b,c in A have a+b, a+c, b+c in A.
    # And no a,b in A have a+b in B (if B is fixed) causing issues with c in A.
    # Actually, we check validity of the base set A U B.
    
    rng = random.Random(42)
    
    # Define a range of L_B and A parameters
    # L_B from 60 to 70
    # L_A from 14 to 20, R_A from L_A to L_B-2
    
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        
        # Randomize around the known good configuration
        # Or grid search if fast enough
        # Grid search:
        L_B = rng.randint(60, 70)
        L_A = rng.randint(14, 20)
        R_A = rng.randint(L_A, min(L_A + 20, L_B - 2))
        
        A = set(range(L_A, R_A + 1))
        B = set(range(L_B, N + 1))
        
        if A & B:
            continue
            
        base = A | B
        
        # Check if base is valid
        if not _is_valid_set(base):
            continue
            
        # Gap candidates: elements not in base
        # Focus on the gap between A and B, and maybe below A
        # Elements < L_A might be risky but let's include them if time permits
        # For now, just the gap (R_A, L_B)
        gap = set(range(R_A + 1, L_B))
        
        # Also consider elements < L_A?
        # If we add x < L_A, sums x+a might land in A or Gap.
        # Let's include 1..L_A-1 in candidates but lower priority?
        # Just use all remaining elements.
        candidates = set(range(1, N + 1)) - base
        
        # Greedy fill
        new_set = _greedy_fill_gap(base, candidates, deadline)
        
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = new_set
            
        # Local improvement: try to swap
        # If we have time, try to remove one element and add two?
        # Or just try to add more in different random orders?
        # The greedy already shuffles.
        
        # If we hit a high score, we can break early?
        if best_size >= 85:
            break

    # Final check
    if not _is_valid_set(best_set):
        # Fallback to known valid set
        best_set = set(range(65, 129))
        
    return [int(x) for x in best_set]
