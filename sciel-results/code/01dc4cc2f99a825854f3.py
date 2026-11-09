"""
Optimized search for additive bases with few representations.

Uses a structural shift from pure greedy peeling to a hybrid approach:
1. Fast numpy-based counting for evaluation.
2. Multi-start local search with simulated annealing-like acceptance.
3. Smart initialization based on interval subsets.
"""

import time
import random
import sys
import numpy as np

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1  # 129

def _get_counts_numpy(A_list):
    """
    Efficiently compute representation counts for sums in [_LO, _HI].
    Uses numpy convolution or matrix operations for speed.
    Returns a numpy array of counts, or None if A is empty.
    """
    if not A_list:
        return None
    
    # Create a boolean array for A
    A_bool = np.zeros(_N + 1, dtype=np.int8)
    for x in A_list:
        if 0 <= x <= _N:
            A_bool[x] = 1
    
    # Convolution gives counts of ordered pairs (a, b) such that a+b = s
    # We need unordered pairs with a <= b.
    # Let C[s] be the count of ordered pairs.
    # For s even, (s/2, s/2) is counted once in ordered, but is 1 unordered.
    # For s odd, all pairs are distinct (a,b) and (b,a), so unordered count is C[s] // 2.
    # For s even, unordered count is (C[s] - 1) // 2 + 1 = (C[s] + 1) // 2.
    # Wait, let's verify.
    # Ordered pairs (a,b) with a,b in A.
    # Unordered pairs {a,b} with a<=b.
    # If a != b, it contributes 2 to ordered count (a,b) and (b,a), and 1 to unordered.
    # If a == b, it contributes 1 to ordered count, and 1 to unordered.
    # Let U[s] be unordered count.
    # C[s] = 2 * (number of pairs with a != b) + (number of pairs with a == b)
    # U[s] = (number of pairs with a != b) + (number of pairs with a == b)
    # So C[s] = 2 * (U[s] - D[s]) + D[s] = 2*U[s] - D[s], where D[s] is 1 if s/2 in A and s even, else 0.
    # Thus U[s] = (C[s] + D[s]) / 2.
    
    # Using FFT convolution for speed
    # A_bool is length 129.
    # Convolution length will be 257.
    # We only need indices 128 to 256.
    
    # numpy.convolve is O(N^2) for small N, but let's try.
    # For N=129, N^2 is ~16k, which is fast.
    # FFT is overkill for this size unless we do many evaluations.
    # However, we are doing many evaluations.
    # Let's stick to numpy.dot or similar if possible, or just the loop if it's fast enough.
    # Actually, np.convolve is implemented in C and is quite fast for 129 elements.
    
    conv = np.convolve(A_bool, A_bool, mode='full')
    
    # Extract the relevant part
    # conv[k] is the number of pairs (i, j) such that i+j = k.
    # We want k in [128, 256].
    counts_ordered = conv[_LO:_HI+1]
    
    # Convert to unordered
    counts_unordered = np.zeros(_RANGE, dtype=np.int32)
    for i in range(_RANGE):
        s = _LO + i
        if s % 2 == 0:
            half = s // 2
            if 0 <= half <= _N and A_bool[half] == 1:
                counts_unordered[i] = (counts_ordered[i] + 1) // 2
            else:
                counts_unordered[i] = counts_ordered[i] // 2
        else:
            counts_unordered[i] = counts_ordered[i] // 2
            
    return counts_unordered

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return np.all(counts >= 1)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return int(np.max(counts))

def _local_search(A_init, rng, deadline, current_best_peak, current_best_A):
    """
    Perform local search starting from A_init.
    Returns best_A, best_peak found in this run.
    """
    A = set(A_init)
    if not A:
        return A_init, float('inf')
    
    counts = _get_counts_numpy(A)
    if not _is_valid(counts):
        # Try to repair or just return invalid? 
        # Better to return a valid set if possible, but local search assumes valid start.
        # If start is invalid, we should probably not use it.
        # But we will ensure starts are valid.
        pass
        
    peak = _get_peak(counts)
    best_A = A.copy()
    best_peak = peak
    
    # If this is already better than global best, update
    if peak < current_best_peak:
        current_best_peak = peak
        current_best_A = A.copy()
        
    # Local search loop
    in_A = list(A)
    not_in_A = [x for x in range(_N + 1) if x not in A]
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Choose a move: swap or add/remove
        # Swap is generally better for maintaining size
        if not in_A or not not_in_A:
            break
            
        x = rng.choice(in_A)
        y = rng.choice(not_in_A)
        
        new_A = A.copy()
        new_A.discard(x)
        new_A.add(y)
        
        new_counts = _get_counts_numpy(new_A)
        if not _is_valid(new_counts):
            continue
            
        new_peak = _get_peak(new_counts)
        
        # Acceptance criteria
        if new_peak < peak:
            A = new_A
            counts = new_counts
            peak = new_peak
            in_A = list(A)
            not_in_A = [x for x in range(_N + 1) if x not in A]
            
            if peak < current_best_peak:
                current_best_peak = peak
                current_best_A = A.copy()
        elif rng.random() < temp:
            # Accept worse solution with some probability (simulated annealing)
            A = new_A
            counts = new_counts
            peak = new_peak
            in_A = list(A)
            not_in_A = [x for x in range(_N + 1) if x not in A]
            
        temp *= cooling
        if temp < 0.01:
            break
            
    return current_best_A, current_best_peak

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # Safety margin
    
    rng = random.Random(42)
    
    best_peak = float('inf')
    best_A = set()
    
    # 1. Greedy Peeling from full set [0, 128]
    # This provides a good baseline.
    A_init = set(range(0, _N + 1))
    A_curr = A_init.copy()
    counts = _get_counts_numpy(A_curr)
    
    # Simple greedy peel
    while time.time() < DEADLINE - 10:
        if not _is_valid(counts):
            break
        peak = _get_peak(counts)
        
        # Find element to remove
        best_removal = None
        best_new_peak = peak
        best_new_size = len(A_curr)
        
        A_list = sorted(A_curr)
        for x in A_list:
            new_A = A_curr.copy()
            new_A.discard(x)
            if not new_A:
                continue
            new_counts = _get_counts_numpy(new_A)
            if not _is_valid(new_counts):
                continue
            new_peak = _get_peak(new_counts)
            
            # Prefer lower peak, then smaller size
            if new_peak < best_new_peak or (new_peak == best_new_peak and len(new_A) < best_new_size):
                best_new_peak = new_peak
                best_new_size = len(new_A)
                best_removal = new_A
                
        if best_removal is not None:
            A_curr = best_removal
            counts = _get_counts_numpy(A_curr)
            if peak < best_peak:
                best_peak = peak
                best_A = A_curr.copy()
        else:
            break
            
    if peak < best_peak:
        best_peak = peak
        best_A = A_curr.copy()
        
    # 2. Local Search with multiple restarts
    # Start from the result of greedy peeling, and also from other random valid sets.
    
    # Restart 1: From greedy result
    if time.time() < DEADLINE - 5:
        best_A, best_peak = _local_search(best_A, rng, DEADLINE - 2, best_peak, best_A)
        
    # Restart 2: From a random subset of [0, 128] that is likely valid
    # We can generate random sets and check validity.
    for _ in range(10):
        if time.time() > DEADLINE - 5:
            break
            
        # Generate a random set of size ~50-70
        size = rng.randint(40, 80)
        A_rand = set(rng.sample(range(_N + 1), size))
        
        counts = _get_counts_numpy(A_rand)
        if _is_valid(counts):
            peak = _get_peak(counts)
            if peak < best_peak:
                best_peak = peak
                best_A = A_rand.copy()
                
            # Run local search on this valid random set
            best_A, best_peak = _local_search(A_rand, rng, DEADLINE - 2, best_peak, best_A)
            
    # Restart 3: Structured sets like intervals
    # Try [k, 128] for various k
    for k in range(0, 50, 5):
        if time.time() > DEADLINE - 5:
            break
        A_int = set(range(k, _N + 1))
        counts = _get_counts_numpy(A_int)
        if _is_valid(counts):
            peak = _get_peak(counts)
            if peak < best_peak:
                best_peak = peak
                best_A = A_int.copy()
            # Local search
            best_A, best_peak = _local_search(A_int, rng, DEADLINE - 2, best_peak, best_A)
            
    # Final check: ensure best_A is valid
    if not best_A:
        # Fallback to seed
        best_A = set(range(64, 129))
        
    final_counts = _get_counts_numpy(best_A)
    if not _is_valid(final_counts):
        # If somehow invalid, fallback
        best_A = set(range(64, 129))
        
    return sorted(best_A)

# EVOLVE-BLOCK-END
