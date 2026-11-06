"""Large reciprocal-subsum-free sets: Greedy + local search construction."""
import time
import math
import sys
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0  # Safety margin under 90 s wall-clock

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

# Precompute weights
_L, _W = None, None

def _init_weights():
    global _L, _W
    if _W is not None:
        return
    L = 1
    for k in range(1, 25):
        L = L * k // math.gcd(L, k)
    _L = L
    _W = [L // k for k in range(1, 25)]

_init_weights()

def _is_subset_sum(target, items):
    """Check if target can be formed by a subset sum of items.
    Uses meet-in-the-middle for efficiency."""
    if target < 0:
        return False
    if target == 0:
        return True  # Empty subset sums to 0

    n = len(items)
    if n == 0:
        return target == 0

    # If target is larger than sum of all items, impossible
    if sum(items) < target:
        return False

    mid = n // 2
    left = items[:mid]
    right = items[mid:]

    left_sums = set()
    # Generate subset sums of left
    for mask in range(1 << len(left)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += left[b]
            mm ^= lb
        left_sums.add(s)

    # Check right subsets
    for mask in range(1 << len(right)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += right[b]
            mm ^= lb
        if s > target:
            continue
        if (target - s) in left_sums:
            return True

    return False

def _is_valid_addition(k_idx, selected_indices):
    """
    Check if adding element k_idx (0-based) to selected_indices (list of 0-based indices)
    results in a valid set.
    selected_indices does NOT contain k_idx.
    """
    S = selected_indices
    # 1. Check if k_idx is subsumed by S
    if S:
        weights_S = [_W[i] for i in S]
        if _is_subset_sum(_W[k_idx], weights_S):
            return False

    # 2. Check if any y in S becomes subsumed by S U {k_idx}
    for y in S:
        diff = _W[y] - _W[k_idx]
        if diff >= 0:
            others = [i for i in S if i != y]
            if others:
                weights_others = [_W[i] for i in others]
                if _is_subset_sum(diff, weights_others):
                    return False
            else:
                if diff == 0:
                    return False

    return True

def _is_set_valid(selected_indices):
    """Check if the entire set is valid (no element is subsumed by others)."""
    S = selected_indices
    for i, y in enumerate(S):
        others = S[:i] + S[i+1:]
        if not others:
            continue
        weights_others = [_W[j] for j in others]
        if _is_subset_sum(_W[y], weights_others):
            return False
    return True

def _try_add_elements(selected):
    """Try to add any remaining elements to the set."""
    improved = True
    while improved:
        if _time_left() < 1.0:
            break
        improved = False
        for k in range(24, 0, -1):
            k_idx = k - 1
            if k_idx in selected:
                continue
            if _is_valid_addition(k_idx, selected):
                selected.append(k_idx)
                improved = True
                break
    return selected

def _try_swap_improve(selected):
    """Try swapping one element out for another to allow more additions."""
    selected = list(selected)
    improved = True
    while improved:
        if _time_left() < 1.0:
            break
        improved = False
        # Try each pair (remove r, add a)
        for r_idx in range(len(selected)):
            for a in range(24, 0, -1):
                a_idx = a - 1
                if a_idx in selected:
                    continue
                # Try swap: remove selected[r_idx], add a_idx
                new_selected = [x for i, x in enumerate(selected) if i != r_idx]
                if _is_valid_addition(a_idx, new_selected):
                    new_selected.append(a_idx)
                    # Try to add more
                    new_selected = _try_add_elements(new_selected)
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        improved = True
                        break
            if improved:
                break
    return selected

def _try_multi_swap(selected):
    """Try removing 1-2 elements and adding 1-2 others."""
    selected = list(selected)
    best = selected
    if _time_left() < 2.0:
        return best

    n = len(selected)
    # Try removing 1 element, adding 1-2
    for r in range(n):
        if _time_left() < 1.0:
            break
        base = [x for i, x in enumerate(selected) if i != r]
        # Add 1
        for a in range(24, 0, -1):
            a_idx = a - 1
            if a_idx in base:
                continue
            if _is_valid_addition(a_idx, base):
                cand = base + [a_idx]
                cand = _try_add_elements(cand)
                if len(cand) > len(best):
                    best = cand
        # Add 2
        for a1 in range(24, 0, -1):
            a1_idx = a1 - 1
            if a1_idx in base:
                continue
            if not _is_valid_addition(a1_idx, base):
                continue
            base2 = base + [a1_idx]
            for a2 in range(a1 - 1, 0, -1):
                a2_idx = a2 - 1
                if a2_idx in base2:
                    continue
                if _is_valid_addition(a2_idx, base2):
                    cand = base2 + [a2_idx]
                    cand = _try_add_elements(cand)
                    if len(cand) > len(best):
                        best = cand
    return best

def construct():
    _init_weights()
    selected = []

    # Greedy descending: try 24, 23, ..., 1
    for k in range(24, 0, -1):
        k_idx = k - 1
        if _time_left() < 1.0:
            break
        if _is_valid_addition(k_idx, selected):
            selected.append(k_idx)

    if not selected:
        return [24]

    # Try to add more elements
    selected = _try_add_elements(selected)

    # Try swap improvements
    if _time_left() > 3.0:
        swapped = _try_swap_improve(selected)
        if len(swapped) > len(selected):
            selected = swapped

    # Try multi-swap
    if _time_left() > 5.0:
        multi = _try_multi_swap(selected)
        if len(multi) > len(selected):
            selected = multi

    # Final add pass
    selected = _try_add_elements(selected)

    return [i + 1 for i in selected]
