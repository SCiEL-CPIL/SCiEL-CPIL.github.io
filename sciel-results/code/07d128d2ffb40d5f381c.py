"""Rewritten program: vectorized greedy + local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_valid_array(arr):
    """
    Check if array of distinct positive ints is valid:
    no element divides the sum of any 3 distinct OTHER elements.
    """
    n = len(arr)
    if n < 4:
        return True
    a = np.asarray(arr, dtype=np.int64)
    # For each element a[i], consider all triples of distinct other indices
    # We need: for all i, for all (j,k,l) distinct from i and each other:
    #   (a[j]+a[k]+a[l]) % a[i] != 0
    
    # Vectorized approach:
    # Sums of all triples: shape (n, n, n) would be too big for large n.
    # Instead, for each i, compute all triple sums of others.
    
    for i in range(n):
        others = np.delete(a, i)
        m = len(others)
        if m < 3:
            continue
        # Generate all combinations of 3 from others
        # Use itertools for triple generation (m choose 3)
        from itertools import combinations
        bad = False
        ai = a[i]
        for j, k, l in combinations(range(m), 3):
            s = others[j] + others[k] + others[l]
            if s % ai == 0:
                bad = True
                break
        if bad:
            return False
    return True

def _is_valid_fast(arr):
    """
    Faster validity check using vectorized triple-sum computation.
    For each element, compute all triple sums of others and check divisibility.
    """
    n = len(arr)
    if n < 4:
        return True
    a = np.asarray(arr, dtype=np.int64)
    
    for i in range(n):
        others = np.delete(a, i)
        m = len(others)
        if m < 3:
            continue
        ai = a[i]
        # Compute all triple sums efficiently
        # For m up to ~48, C(48,3)=17296, which is manageable
        # Use numpy broadcasting for pairs, then add third
        # Actually, let's use a simpler approach: iterate but with early exit
        # For speed, precompute pair sums
        # pairs: C(m,2) sums
        idx = np.arange(m)
        # Create all pairs
        i_idx, j_idx = np.triu_indices(m, k=1)
        pair_sums = others[i_idx] + others[j_idx]
        # For each pair, add each remaining element
        # This is O(m^3) but in C
        # Alternative: for each k, sum over all pairs not involving k
        # Simpler: just loop over k and check
        for k in range(m):
            # Triples including k: need two others from [0..m-1] excluding k
            mask = idx != k
            remaining = others[mask]
            # All pairs from remaining
            if len(remaining) < 2:
                continue
            r_idx = np.arange(len(remaining))
            p_i, p_j = np.triu_indices(len(remaining), k=1)
            ps = remaining[p_i] + remaining[p_j]
            # Check if any (ps + others[k]) % ai == 0
            if np.any((ps + others[k]) % ai == 0):
                return False
    return True

def _greedy_random(seed=0):
    """Greedy construction with random ordering."""
    rng = np.random.default_rng(seed)
    candidates = np.arange(1, 193, dtype=np.int64)
    rng.shuffle(candidates)
    selected = []
    for x in candidates:
        test = selected + [int(x)]
        if _is_valid_fast(test):
            selected.append(int(x))
    return selected

def _greedy_large_first():
    """Greedy construction preferring larger numbers."""
    selected = []
    for x in range(192, 0, -1):
        test = selected + [x]
        if _is_valid_fast(test):
            selected.append(x)
    return selected

def _local_search(arr, time_limit):
    """
    Local search: try to add elements, swap elements to improve cardinality.
    Returns improved array.
    """
    arr = list(arr)
    start = time.time()
    best = len(arr)
    best_arr = list(arr)
    
    # Try to add elements from 1..192 not in arr
    available = [x for x in range(1, 193) if x not in set(arr)]
    
    # Try adding
    for x in available:
        if time.time() - start > time_limit - 5:
            break
        test = arr + [x]
        if _is_valid_fast(test):
            arr = test
            best = len(arr)
            best_arr = list(arr)
            # Update available
            available.remove(x)
    
    # Try swaps: remove one, add one
    for _ in range(100):  # limited iterations
        if time.time() - start > time_limit - 5:
            break
        improved = False
        for i in range(len(arr)):
            if time.time() - start > time_limit - 5:
                break
            for x in available:
                if time.time() - start > time_limit - 5:
                    break
                test = [v for j, v in enumerate(arr) if j != i] + [x]
                if _is_valid_fast(test):
                    arr = test
                    available.remove(x)
                    available.append(arr[i])
                    best = len(arr)
                    best_arr = list(arr)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_arr

def construct():
    deadline = time.time() + 55  # 55s safety margin within 60s
    
    # Start with greedy large-first
    arr = _greedy_large_first()
    best_arr = list(arr)
    best_len = len(arr)
    
    # Try multiple random greedy seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        candidate = _greedy_random(seed)
        if len(candidate) > best_len:
            best_arr = candidate
            best_len = len(candidate)
    
    # Local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best_arr, deadline - time.time())
        if len(improved) > best_len:
            best_arr = improved
            best_len = len(improved)
    
    # Final check: ensure validity
    if not _is_valid_fast(best_arr):
        # Fallback to a known valid set
        best_arr = _greedy_large_first()
    
    return best_arr
# EVOLVE-BLOCK-END
