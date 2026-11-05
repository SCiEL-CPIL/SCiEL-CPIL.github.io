"""Fast integer-based meet-in-the-middle for reciprocal subset-sum checks.
Uses LCM-scaled integer arithmetic to avoid slow Fraction operations."""
import time
import sys
import math
import random
from fractions import Fraction

# --- Precomputation with integer arithmetic ---
# LCM of 1..24
_LCM_24 = 5354228880  # lcm(1,2,...,24)

# For each element a (1..24), 1/a scaled by LCM_24:
# rec[a] = LCM_24 // a  (integer)
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
_t1 = time.time()

# Build violation database: for each target a (1..24), store all (lmask, rmask) pairs
# where lmask is a mask into _LEFT (12 bits) and rmask is a mask into _RIGHT (12 bits)
# such that sum_left(lmask) + sum_right(rmask) == _REC[a], and not both zero.
# We store as list of (lmask, rmask) tuples.

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

# For faster lookup, also build: for each a, a set of (lmask, rmask) for O(1) membership?
# Actually we iterate, so list is fine. But let's also build a quick check structure.
# For each a and each lmask, store the set of rmask values that work.
_VIOLATION_BY_LMASK = {}  # a -> {lmask: set(rmask)}
for a in range(1, 25):
    d = {}
    for lm, rm in _VIOLATION_DB[a]:
        if lm not in d:
            d[lm] = set()
        d[lm].add(rm)
    _VIOLATION_BY_LMASK[a] = d

def _check_safe(a_idx, S_mask):
    """Check if element a_idx (0-based, element = a_idx+1) in set S_mask is safe.
    S_mask: 24-bit int where bit i means element (i+1) is in S.
    Returns True if no nonempty subset of S \\ {a_idx} sums to 1/(a_idx+1).
    """
    val = a_idx + 1
    avail_mask = S_mask & ~(1 << a_idx)
    
    if avail_mask == 0:
        return True
    
    left_avail = avail_mask & 0xFFF       # bits 0..11 (elements 1..12)
    right_avail = (avail_mask >> 12) & 0xFFF  # bits 12..23 mapped to 0..11 (elements 13..24)
    
    pairs = _VIOLATION_DB.get(val, [])
    if not pairs:
        return True
    
    for lm, rm in pairs:
        if (lm & left_avail) == lm and (rm & right_avail) == rm:
            return False
    return True

def _check_all_safe(S_mask):
    """Check all elements in S_mask are safe."""
    for i in range(24):
        if S_mask >> i & 1:
            if not _check_safe(i, S_mask):
                return False
    return True

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality such that no reciprocal
    of a selected element equals the sum of reciprocals of a subset of the other selected elements."""
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    
    best_set = 0
    best_size = 0
    
    # Phase 1: Deterministic greedy with multiple orderings
    # Try several orderings and take the best result
    
    orderings = [
        list(range(24)),                          # ascending: 1,2,...,24
        list(range(23, -1, -1)),                  # descending: 24,23,...,1
        # Odd then even
        [i for i in range(24) if i % 2 == 0] + [i for i in range(24) if i % 2 == 1],
        # Even then odd
        [i for i in range(24) if i % 2 == 1] + [i for i in range(24) if i % 2 == 0],
        # Large reciprocals first (small elements first) - already ascending
        # Try: primes first
        [i for i in range(24) if i+1 in (2,3,5,7,11,13,17,19,23)] + 
        [i for i in range(24) if i+1 not in (2,3,5,7,11,13,17,19,23)],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = 0
        for idx in order:
            trial = S | (1 << idx)
            if _check_all_safe(trial):
                S = trial
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S
    
    # Phase 2: Local search - try to add more elements
    # Repeatedly try adding the best element that fits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(24):
            if time.time() > deadline:
                break
            if best_set >> idx & 1:
                continue
            trial = best_set | (1 << idx)
            if _check_all_safe(trial):
                best_set = trial
                best_size += 1
                improved = True
    
    # Phase 3: Randomized local search with restarts
    # Try random subsets and improve them
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Start with a random subset
        S = 0
        for idx in range(24):
            if random.random() < 0.5:
                S |= (1 << idx)
        # Remove violations
        S = _remove_violations(S)
        # Try to add more
        for idx in range(24):
            if time.time() > deadline:
                break
            if S >> idx & 1:
                continue
            trial = S | (1 << idx)
            if _check_all_safe(trial):
                S = trial
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S
    
    # Phase 4: Try to swap elements to increase size
    # For each pair (in, out), try swapping
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for out_idx in range(24):
                if time.time() > deadline:
                    break
                if not (best_set >> out_idx & 1):
                    continue
                for in_idx in range(24):
                    if time.time() > deadline:
                        break
                    if best_set >> in_idx & 1:
                        continue
                    # Try: remove out_idx, add in_idx
                    trial = best_set & ~(1 << out_idx) | (1 << in_idx)
                    if _check_all_safe(trial):
                        # Check if this helps: we need to be able to add more after
                        # For now, just accept if same size (might enable future additions)
                        # Actually, only accept if it leads to improvement
                        # Let's just try it and see if we can add more
                        S2 = trial
                        added = False
                        for add_idx in range(24):
                            if time.time() > deadline:
                                break
                            if S2 >> add_idx & 1:
                                continue
                            trial2 = S2 | (1 << add_idx)
                            if _check_all_safe(trial2):
                                S2 = trial2
                                added = True
                        if S2.bit_count() > best_size:
                            best_set = S2
                            best_size = S2.bit_count()
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Convert best_set to list
    if best_set:
        result = [i + 1 for i in range(24) if best_set >> i & 1]
    else:
        result = [1]
    
    if not result:
        result = [1]
    
    return result

def _remove_violations(S_mask):
    """Remove elements from S_mask until no violations remain.
    Greedy: remove the element that causes the most violations."""
    while S_mask:
        if _check_all_safe(S_mask):
            break
        # Find which elements are violating
        violators = []
        for i in range(24):
            if S_mask >> i & 1:
                if not _check_safe(i, S_mask):
                    violators.append(i)
        if not violators:
            break
        # Remove the first violator (or the one with smallest value, since small elements
        # have large reciprocals and are more likely to be targets)
        # Actually, remove the one that's causing issues
        # Heuristic: remove the smallest-index violator
        to_remove = min(violators)
        S_mask &= ~(1 << to_remove)
    return S_mask

# Test
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
