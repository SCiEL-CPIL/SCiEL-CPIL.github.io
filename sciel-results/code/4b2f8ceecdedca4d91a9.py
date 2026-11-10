"""Rewritten program: local search with NumPy-vectorized AP discrepancy."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin before 90s kill


def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [1, n]. signs is a 1D numpy array of int8 (-1 or +1), length n.
    Returns the max absolute sum (an integer).
    """
    n = len(signs)
    max_disc = 1  # singletons give |sum| = 1

    # For each step d from 1 to n-1, and each residue r in [0, d),
    # the APs with step d starting at positions r, r+d, r+2d, ... 
    # are contiguous subsequences of signs[r::d].
    # We compute the max absolute subarray sum for each such sequence.
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            # Kadane's algorithm for max absolute subarray sum
            # We need max over all contiguous subarrays of |sum|
            # This equals max(max_subarray_sum, -min_subarray_sum)
            # where max_subarray_sum is the max sum of any contiguous subarray
            # and min_subarray_sum is the min sum.
            
            # Use a vectorized approach: compute all prefix sums, then
            # max_subarray = max(prefix[i] - min(prefix[0..i-1]))
            # min_subarray = min(prefix[i] - max(prefix[0..i-1]))
            
            csum = np.cumsum(seq)
            # For each position i, the best subarray ending at i has sum:
            #   csum[i] - min(csum[0..i-1])  for max
            #   csum[i] - max(csum[0..i-1])  for min
            # We need to handle the case where the subarray starts at 0.
            
            # Efficient approach: use the fact that for a sequence of -1/+1,
            # the max absolute subarray sum can be computed incrementally.
            # But let's use a simpler O(L) approach per sequence.
            
            # Actually, for a binary ±1 sequence, we can compute:
            # max_abs = max over all contiguous subarrays of |sum|
            # This is equivalent to: for the prefix sum P[0]=0, P[i]=sum_{j=1}^{i} seq[j],
            # the max subarray sum = max_{i>j} (P[i]-P[j]), min subarray sum = min_{i>j} (P[i]-P[j])
            # max_abs = max(max_subarray_sum, -min_subarray_sum)
            
            # Compute using running min/max of prefix sums
            prefix = np.empty(len(seq) + 1, dtype=np.int32)
            prefix[0] = 0
            prefix[1:] = csum
            
            # max_subarray: for each i from 1 to L, P[i] - min(P[0..i-1])
            running_min = np.empty(len(prefix), dtype=np.int32)
            running_min[0] = prefix[0]
            for i in range(1, len(prefix)):
                running_min[i] = min(running_min[i-1], prefix[i])
            # But we need min of P[0..i-1], not P[0..i]
            # Let's redo: for i from 1 to L, best ending at i is P[i] - min(P[0..i-1])
            
            # More carefully:
            # min_prefix_before[i] = min(P[0], P[1], ..., P[i-1]) for i >= 1
            # max_prefix_before[i] = max(P[0], P[1], ..., P[i-1]) for i >= 1
            
            min_before = np.empty(len(prefix), dtype=np.int32)
            max_before = np.empty(len(prefix), dtype=np.int32)
            min_before[0] = 0  # unused
            max_before[0] = 0  # unused
            for i in range(1, len(prefix)):
                min_before[i] = min(min_before[i-1], prefix[i-1])
                max_before[i] = max(max_before[i-1], prefix[i-1])
            
            max_subarray = np.max(prefix[1:] - min_before[1:])
            min_subarray = np.min(prefix[1:] - max_before[1:])
            
            disc = max(max_subarray, -min_subarray)
            if disc > max_disc:
                max_disc = int(disc)
    
    return int(max_disc)


def _max_abs_ap_sum_fast(signs):
    """
    Faster version using a different approach: iterate over steps and use
    vectorized cumsum with a smarter Kadane-like computation.
    """
    n = len(signs)
    max_disc = 1

    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            
            # For a ±1 sequence, the max absolute contiguous subarray sum
            # can be found by tracking the running sum and the min/max of 
            # prefix sums seen so far.
            # Let's do it in a loop but vectorize where possible.
            
            # Actually, let's just use a simple O(L) loop since L is small for large d
            # and for small d, the number of sequences is small.
            
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            local_max = 0
            local_min = 0
            
            for val in seq:
                prefix += val
                # Max subarray ending here: prefix - min_prefix
                # Min subarray ending here: prefix - max_prefix
                if prefix - min_prefix > local_max:
                    local_max = prefix - min_prefix
                if prefix - max_prefix < local_min:
                    local_min = prefix - max_prefix
                
                if prefix < min_prefix:
                    min_prefix = prefix
                if prefix > max_prefix:
                    max_prefix = prefix
            
            disc = max(local_max, -local_min)
            if disc > max_disc:
                max_disc = int(disc)
    
    return int(max_disc)


def _greedy_construct():
    """
    Greedy construction: at each position, choose the sign that minimizes
    the current maximum AP discrepancy.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        best_sign = 1
        best_disc = 1000000
        
        for s in (1, -1):
            signs[i] = s
            disc = _max_abs_ap_sum_fast(signs)
            if disc < best_disc:
                best_disc = disc
                best_sign = s
        
        signs[i] = best_sign
    
    return signs, best_disc


def _local_search(signs, max_iter=50000):
    """
    First-improvement hill climbing with random perturbations.
    """
    n = _N
    signs = signs.copy()
    current_disc = _max_abs_ap_sum_fast(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Pick a random position to flip
        idx = rng.randrange(n)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        new_disc = _max_abs_ap_sum_fast(signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[idx] = old_val
    
    return best_signs, best_disc


def _local_search_with_restarts(num_restarts=10, max_iter_per_restart=20000):
    """
    Multiple restarts of local search with random initializations.
    """
    best_signs = None
    best_disc = 1000000
    
    rng = random.Random(123)
    
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Random initialization
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(_N)], dtype=np.int8)
        
        signs, disc = _local_search(signs, max_iter=max_iter_per_restart)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs
    
    return best_signs, best_disc


def construct():
    start_time = time.time()
    
    # Start with greedy construction
    signs, disc = _greedy_construct()
    best_signs = signs.copy()
    best_disc = disc
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 10.0:
        return [int(s) for s in best_signs]
    
    # Local search from greedy
    signs, disc = _local_search(best_signs, max_iter=30000)
    if disc < best_disc:
        best_disc = disc
        best_signs = signs.copy()
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 10.0:
        return [int(s) for s in best_signs]
    
    # Local search with random restarts
    signs, disc = _local_search_with_restarts(num_restarts=5, max_iter_per_restart=15000)
    if disc < best_disc:
        best_disc = disc
        best_signs = signs.copy()
    
    return [int(s) for s in best_signs]
# EVOLVE-BLOCK-END
