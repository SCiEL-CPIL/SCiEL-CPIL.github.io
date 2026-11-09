"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import numpy as np

_LCM = 232792560

def _find_collision_fast(subset):
    """Find two different subsets with the same sum, or None if dissociated.
    Returns (mask1, mask2) where mask1 < mask2."""
    k = len(subset)
    if k == 0:
        return None
    scaled = np.array([_LCM // x for x in subset], dtype=np.int64)
    n = 1 << k
    
    # Efficient vectorized subset sum computation
    # sums[mask] = sum of scaled[i] for each i where mask has bit i set
    # We can compute this using the property that for each bit position,
    # we add scaled[i] to all masks that have that bit set
    sums = np.zeros(n, dtype=np.int64)
    for i in range(k):
        bit = 1 << i
        # Masks where bit i is set: blocks of size bit, every other block
        # Pattern: [0..bit-1] no, [bit..2*bit-1] yes, [2*bit..3*bit-1] no, etc.
        # So masks with bit i set are: bit + j*(2*bit) for j in range, plus offsets 0..bit-1
        # More simply: use the fact that (mask & bit) != 0
        # Create a mask array of all masks
        masks = np.arange(n, dtype=np.int64)
        # For each block of size 2*bit, the second half has bit i set
        # Block j: [j*2*bit, (j+1)*2*bit - 1], second half: [j*2*bit + bit, (j+1)*2*bit - 1]
        # This is equivalent to: (mask >> i) & 1 == 1
        # Vectorized: create indices where bit i is set
        # For bit i, the pattern repeats every 2*bit: first bit values have 0, next bit values have 1
        # So: indices where (idx // bit) % 2 == 1, i.e., idx in [bit, 2*bit-1], [3*bit, 4*bit-1], etc.
        # Let's use a simpler approach: create a boolean mask
        bit_mask = (masks & bit) != 0
        sums[bit_mask] += scaled[i]
    
    # Check for collisions using numpy
    sorted_sums, sort_indices = np.sort(sums), np.argsort(sums)
    diffs = np.diff(sorted_sums)
    collision_pos = np.where(diffs == 0)[0]
    if len(collision_pos) > 0:
        pos = collision_pos[0]
        mask1 = int(sort_indices[pos])
        mask2 = int(sort_indices[pos + 1])
        return min(mask1, mask2), max(mask1, mask2)
    return None

def _greedy_remove(subset, heuristic='largest', deadline=None):
    """Start with subset, iteratively remove elements causing collisions."""
    current = list(subset)
    while True:
        if deadline and time.time() > deadline:
            break
        col = _find_collision_fast(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        if heuristic == 'largest':
            to_remove = max(sym_diff, key=lambda i: current[i])
        elif heuristic == 'smallest':
            to_remove = min(sym_diff, key=lambda i: current[i])
        else:
            to_remove = random.choice(sym_diff)
        current.remove(current[to_remove])
    return current

def _try_add_elements(best, removed, deadline):
    """Try to add back removed elements one by one."""
    current = list(best)
    for x in removed:
        if time.time() > deadline:
            break
        candidate = current + [x]
        candidate.sort()
        if _find_collision_fast(candidate) is None:
            current = candidate
    return current

def _try_swaps(best, not_in_set, deadline):
    """Try swapping elements in best with elements not in best."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(current):
            if time.time() > deadline:
                break
            for b in remaining:
                candidate = current[:]
                candidate[i] = b
                candidate.sort()
                if _find_collision_fast(candidate) is None:
                    current = candidate
                    in_set = set(current)
                    remaining = [x for x in not_in_set if x not in in_set]
                    improved = True
                    break
    return current

def _multi_swap_search(best, not_in_set, deadline):
    """Try removing up to 2 elements and adding up to 2 from not_in_set."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    best_found = list(current)
    
    # Try single swaps first (already done in _try_swaps, but do it here for consistency)
    for i, a in enumerate(current):
        if time.time() > deadline:
            break
        for b in remaining:
            candidate = current[:]
            candidate[i] = b
            candidate.sort()
            if _find_collision_fast(candidate) is None:
                if len(candidate) > len(best_found):
                    best_found = candidate
                else:
                    best_found = candidate
                in_set = set(best_found)
                remaining = [x for x in not_in_set if x not in in_set]
                break
    
    # Try double swaps: remove 2, add 2
    # Order remaining elements to try promising ones first
    # Prefer elements that are closer in magnitude to the ones being removed
    if time.time() < deadline - 5:
        n_cur = len(current)
        for i in range(n_cur):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cur):
                if time.time() > deadline:
                    break
                # Sort remaining by proximity to the elements being removed
                removed_vals = [current[i], current[j]]
                remaining_sorted = sorted(remaining, key=lambda x: min(abs(x - r) for r in removed_vals))
                for b1 in remaining_sorted:
                    if time.time() > deadline:
                        break
                    for b2 in remaining_sorted:
                        if b2 == b1:
                            continue
                        candidate = current[:]
                        # Remove j first (higher index) to keep indices stable
                        candidate.pop(j)
                        candidate.pop(i)
                        candidate.extend([b1, b2])
                        candidate.sort()
                        if _find_collision_fast(candidate) is None:
                            if len(candidate) > len(best_found):
                                best_found = candidate
                                in_set = set(best_found)
                                remaining = [x for x in not_in_set if x not in in_set]
                            break
    return best_found

def construct():
    start_time = time.time()
    deadline = start_time + 85
    universe = list(range(1, 23))
    best = []
    best_len = 0
    heuristics = ['largest', 'smallest', 'random', 'largest', 'smallest', 'random', 'largest']
    for h in heuristics:
        if time.time() > deadline - 12:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 12)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 8)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 5)
        if time.time() < deadline - 3:
            result2 = _multi_swap_search(result, not_in_set, deadline - 3)
            if len(result2) > len(result):
                result = result2
        if len(result) > best_len:
            best_len = len(result)
            best = result
    
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 1)
    not_in_set = [x for x in universe if x not in set(best)]
    if time.time() < deadline - 0.5:
        best = _try_swaps(best, not_in_set, deadline - 0.5)
    return best
