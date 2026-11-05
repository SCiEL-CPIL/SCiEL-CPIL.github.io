"""Fast integer-based meet-in-the-middle for reciprocal subset-sum checks.
Uses LCM-scaled integer arithmetic to avoid slow Fraction operations.
Combined with systematic search by cardinality for maximum set size."""
import time
import sys
import math
import random

# --- Precomputation with integer arithmetic ---
_LCM_24 = 5354228880  # lcm(1,2,...,24)

# For each element a (1..24), 1/a scaled by LCM_24:
_REC = [0] + [_LCM_24 // a for a in range(1, 25)]  # index 1..24

# Left half: elements 1..12, Right half: elements 13..24
_LEFT = list(range(1, 13))   # elements 1..12
_RIGHT = list(range(13, 25))  # elements 13..24

def _build_int_subset_sums(arr):
    """Build dict: integer_sum -> list of masks (bitmask into arr) for nonempty subsets.
    Also includes empty subset: sum=0 -> [0]."""
    n = len(arr)
    sums = {0: [0]}  # include empty
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += _REC[arr[i]]
        if s not in sums:
            sums[s] = []
        sums[s].append(mask)
    return sums

_t0 = time.time()
_sums_left = _build_int_subset_sums(_LEFT)
_sums_right = _build_int_subset_sums(_RIGHT)

# Build violation database: for each target a (1..24), store all (lmask, rmask) pairs
# where lmask is a mask into _LEFT (12 bits) and rmask is a mask into _RIGHT (12 bits)
# such that sum_left(lmask) + sum_right(rmask) == _REC[a], and not both zero.
_VIOLATION_DB = {}  # a (1..24) -> list of (lmask, rmask)

for a in range(1, 25):
    target = _REC[a]
    pairs = []
    for sL, l_masks in _sums_left.items():
        needR = target - sL
        if needR < 0:
            continue
        r_masks = _sums_right.get(needR, [])
        for lm in l_masks:
            for rm in r_masks:
                if lm == 0 and rm == 0:
                    continue
                pairs.append((lm, rm))
    _VIOLATION_DB[a] = pairs

def _get_minimal_forbidden_sets():
    """Find all minimal forbidden sets.
    A set F is forbidden if there exists a in F such that 1/a = sum(1/b for b in F\{a}).
    We find all such F and keep only the minimal ones."""
    forbidden_candidates = set()
    
    for a in range(1, 25):
        target = _REC[a]
        # We need subsets of {1..24} \ {a} summing to target.
        # We iterate through all pairs (lm, rm) from _VIOLATION_DB[a]
        # But we must ensure that 'a' is NOT in the subset.
        # The _VIOLATION_DB was built using ALL elements.
        # So we need to filter out pairs where 'a' is included in the mask.
        
        pairs = _VIOLATION_DB[a]
        for lm, rm in pairs:
            # Construct the full 24-bit mask of the subset T
            # lm is bits 0-11 (elements 1-12)
            # rm is bits 0-11 (elements 13-24)
            full_mask_T = lm | (rm << 12)
            
            # Check if 'a' is in T
            # Element a corresponds to bit (a-1)
            if full_mask_T & (1 << (a - 1)):
                continue
            
            # T is a valid subset summing to 1/a, and a is not in T.
            # F = T U {a}
            full_mask_F = full_mask_T | (1 << (a - 1))
            forbidden_candidates.add(full_mask_F)
    
    # Now filter for minimal sets
    # A set F is minimal if no proper subset of F is in forbidden_candidates
    # Since the number of candidates is likely small, we can do this simply.
    minimal_sets = []
    candidates_list = list(forbidden_candidates)
    
    # Sort by size to check smaller ones first?
    # Actually, if F1 is a subset of F2, F1 is minimal, F2 is not.
    # We can iterate and check if any other candidate is a proper subset.
    # To optimize, we can just check all pairs.
    # N is likely < 1000. N^2 is < 1,000,000. Fast.
    
    n_cand = len(candidates_list)
    is_minimal = [True] * n_cand
    
    for i in range(n_cand):
        if not is_minimal[i]:
            continue
        Fi = candidates_list[i]
        for j in range(n_cand):
            if i == j:
                continue
            Fj = candidates_list[j]
            # Check if Fj is a proper subset of Fi
            if Fj != Fi and (Fj & Fi) == Fj:
                is_minimal[i] = False
                break
    
    for i in range(n_cand):
        if is_minimal[i]:
            minimal_sets.append(candidates_list[i])
            
    return minimal_sets

_MIN_FORBIDDEN = _get_minimal_forbidden_sets()

def _contains_forbidden(mask):
    """Check if mask contains any minimal forbidden set."""
    for F in _MIN_FORBIDDEN:
        if (F & mask) == F:
            return True
    return False

def _branch_and_bound(idx, current_mask, current_size, best_size, best_mask):
    """Recursive branch and bound to find largest safe subset."""
    nonlocal best_size, best_mask
    
    # Prune 1: Max possible size
    if current_size + (24 - idx) <= best_size:
        return
    
    # Prune 2: Check if current set is already invalid
    if _contains_forbidden(current_mask):
        return
    
    if idx == 24:
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
        return
    
    # Option 1: Exclude element idx
    _branch_and_bound(idx + 1, current_mask, current_size, best_size, best_mask)
    
    # Option 2: Include element idx
    new_mask = current_mask | (1 << idx)
    # Quick check: does adding this element create a forbidden set immediately?
    # We can just let the recursive call check it, or check now for efficiency.
    if not _contains_forbidden(new_mask):
        _branch_and_bound(idx + 1, new_mask, current_size + 1, best_size, best_mask)
    else:
        # Even if it's forbidden now, maybe removing some other element later?
        # No, branch and bound usually assumes we build the set incrementally.
        # If new_mask is forbidden, we cannot include idx in a valid set that contains current_mask.
        # So we skip this branch.
        pass

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality such that no reciprocal
    of a selected element equals the sum of reciprocals of a subset of the other selected elements."""
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    
    best_size = 0
    best_mask = 0
    
    # Run Branch and Bound
    # We can order elements to improve pruning.
    # Elements that are in many forbidden sets should be considered earlier?
    # Or later?
    # Usually, if we include a "bad" element early, we prune more.
    # Let's count how many minimal forbidden sets each element is in.
    
    degree = [0] * 24
    for F in _MIN_FORBIDDEN:
        for i in range(24):
            if F & (1 << i):
                degree[i] += 1
    
    # Order elements by degree descending
    order = sorted(range(24), key=lambda i: degree[i], reverse=True)
    
    # Remap indices?
    # The B&B function uses idx 0..23 corresponding to elements 1..24.
    # If we change the order, we need to adjust the logic.
    # Let's stick to natural order for simplicity first, or map it.
    # Actually, we can just pass the order to the B&B.
    
    def _bnb(order_idx, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if order_idx == 24:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        elem = order[order_idx]
        
        # Prune
        if current_size + (24 - order_idx) <= best_size:
            return
            
        if _contains_forbidden(current_mask):
            return
        
        # Exclude elem
        _bnb(order_idx + 1, current_mask, current_size)
        
        # Include elem
        new_mask = current_mask | (1 << elem)
        if not _contains_forbidden(new_mask):
            _bnb(order_idx + 1, new_mask, current_size + 1)

    _bnb(0, 0, 0)
    
    # Convert best_mask to list
    if best_mask:
        result = [i + 1 for i in range(24) if best_mask >> i & 1]
    else:
        result = [1]
        
    if not result:
        result = [1]
        
    return result

# Test
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
