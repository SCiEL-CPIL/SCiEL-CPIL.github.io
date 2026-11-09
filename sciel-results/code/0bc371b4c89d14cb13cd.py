import numpy as np
import time
import math
import random

N = 18
TOTAL_INPUTS = 1 << N

def _batcher_bitonic_sort(N):
    """
    Generate Batcher's Bitonic Sort network for N wires.
    """
    if N == 1:
        return []
    
    def _bitonic_merge(N, offset=0, direction=1):
        # direction: 1 for ascending, -1 for descending (not used directly in standard bitonic sort merge, 
        # but the merge assumes one half asc, one desc)
        comps = []
        # Bitonic merge of two sorted halves of size N/2?
        # Standard Bitonic Merge for N:
        # Step 1: Compare (i, i+N/2) for i in 0..N/2-1
        # Step 2: Recursively merge first half (asc) and second half (desc)
        # Actually, the standard algorithm is:
        # 1. For k in log(N)..1:
        #    For i in 0..N - 2^k:
        #       if (i % 2^(k+1)) < 2^k:
        #          compare(i, i + 2^k)
        
        # Let's use the recursive definition which is easier to get right for arbitrary N?
        # No, the iterative one is standard.
        
        # For N=18, we can't use the standard power-of-2 loop directly without handling boundaries.
        # Batcher's Bitonic Sort works for any N by:
        # 1. Split into N1, N2.
        # 2. Sort N1 (asc), Sort N2 (desc).
        # 3. Bitonic Merge N1, N2.
        
        # Let's implement recursive Bitonic Sort.
        pass

    def _batcher_sort_recursive(N, offset=0, direction=1):
        if N == 1:
            return []
        
        N1 = N // 2
        N2 = N - N1
        
        # Sort first half
        comps1 = _batcher_sort_recursive(N1, offset, 1)
        # Sort second half (reverse direction for bitonic merge)
        comps2 = _batcher_sort_recursive(N2, offset + N1, -1)
        
        # Merge
        merge_comps = _bitonic_merge(N, offset, 1)
        
        return comps1 + comps2 + merge_comps

    def _bitonic_merge(N, offset=0, direction=1):
        # Merges two bitonic sequences of length N (offset to offset+N)
        # The first N/2 are ascending, the second N/2 are descending (relative to each other)
        # Result is sorted in 'direction'
        
        if N == 1:
            return []
        
        comps = []
        # Step 1: Compare (i, i + N/2)
        step = N // 2
        for i in range(step):
            i1 = offset + i
            i2 = offset + i + step
            comps.append((i1, i2))
        
        # Step 2: Recursively merge
        # First half (offset .. offset+step) should be sorted in 'direction'
        # Second half (offset+step .. offset+N) should be sorted in 'direction'
        # Wait, after the first step, the first half is bitonic and the second half is bitonic?
        # Yes.
        
        comps1 = _bitonic_merge(step, offset, direction)
        comps2 = _bitonic_merge(N - step, offset + step, direction)
        
        return comps + comps1 + comps2

    return _batcher_sort_recursive(N, 0, 1)

def _verify_network(comps, N, buffer_min, buffer_max, base_matrix):
    """
    Verify a single network.
    Returns True if sorted.
    """
    matrix = base_matrix.copy()
    
    for i, j in comps:
        np.minimum(matrix[:, i], matrix[:, j], out=buffer_min)
        np.maximum(matrix[:, i], matrix[:, j], out=buffer_max)
        matrix[:, i] = buffer_min
        matrix[:, j] = buffer_max
        
    # Check if sorted
    # diff >= 0 for all adjacent pairs
    # We can check if any diff < 0
    # np.diff creates a new array.
    # Alternative: loop? No, np.diff is fast.
    diff = np.diff(matrix, axis=1)
    return bool(np.all(diff >= 0))

def _greedy_prune(comps, N, buffer_min, buffer_max, base_matrix, deadline):
    """
    Greedy pruning: try to remove each comparator if possible.
    """
    best_comps = list(comps)
    i = 0
    while i < len(best_comps) and time.time() < deadline:
        test = best_comps[:i] + best_comps[i+1:]
        if _verify_network(test, N, buffer_min, buffer_max, base_matrix):
            best_comps = test
            # Do not increment i, as indices shifted
        else:
            i += 1
    return best_comps

def _local_search(comps, N, buffer_min, buffer_max, base_matrix, deadline):
    """
    Local search: try removing or swapping/replacing comparators.
    """
    best_comps = list(comps)
    best_count = len(best_comps)
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try random removals
        if len(best_comps) > 0:
            idx = random.randint(0, len(best_comps) - 1)
            test = best_comps[:idx] + best_comps[idx+1:]
            if _verify_network(test, N, buffer_min, buffer_max, base_matrix):
                best_comps = test
                best_count -= 1
                improved = True
                no_improve_count = 0
                continue
        
        # 2. Try random replacement (change endpoints)
        if len(best_comps) > 0:
            idx = random.randint(0, len(best_comps) - 1)
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            
            # Avoid identical comparator
            if best_comps[idx] == (new_i, new_j):
                continue
                
            test = list(best_comps)
            test[idx] = (new_i, new_j)
            
            if _verify_network(test, N, buffer_min, buffer_max, base_matrix):
                best_comps = test
                # Count is same, but structure changed. 
                # We accept to explore.
                improved = True
                no_improve_count = 0
                continue
        
        # 3. Try swapping order of two comparators
        if len(best_comps) > 1:
            idx1 = random.randint(0, len(best_comps) - 1)
            idx2 = random.randint(0, len(best_comps) - 1)
            if idx1 != idx2:
                test = list(best_comps)
                test[idx1], test[idx2] = test[idx2], test[idx1]
                if _verify_network(test, N, buffer_min, buffer_max, base_matrix):
                    best_comps = test
                    improved = True
                    no_improve_count = 0
                    continue

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a more aggressive move: remove a random one and replace with 2? No.
                # Just break or continue with higher threshold?
                break
                
    return best_comps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Pre-allocate buffers
    buffer_min = np.empty(TOTAL_INPUTS, dtype=np.uint8)
    buffer_max = np.empty(TOTAL_INPUTS, dtype=np.uint8)
    
    # Pre-compute base matrix (all inputs)
    # inputs = np.arange(2**N, dtype=np.uint32)
    # bits = 1 << np.arange(N)
    # base_matrix = (inputs[:, None] & bits[None, :]) > 0
    
    # More efficient base matrix generation:
    # We need a matrix of shape (2^N, N) where matrix[k, i] is the i-th bit of k.
    # We can use broadcasting.
    inputs = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    bits = 1 << np.arange(N, dtype=np.uint32)
    base_matrix = (inputs[:, None] & bits[None, :]) > 0
    
    # 1. Generate Seed
    try:
        comps = _batcher_bitonic_sort(N)
        # Deduplicate? Bitonic sort shouldn't have duplicates, but let's be safe.
        # Order matters, so we can't just use set.
        # We can remove consecutive duplicates?
        unique_comps = []
        for c in comps:
            if not unique_comps or unique_comps[-1] != c:
                unique_comps.append(c)
        comps = unique_comps
    except Exception:
        # Fallback to Odd-Even Transposition
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
    
    best_comps = list(comps)
    best_count = len(best_comps)
    
    # 2. Greedy Pruning
    if time.time() < deadline - 20:
        pruned = _greedy_prune(best_comps, N, buffer_min, buffer_max, base_matrix, deadline - 20)
        if len(pruned) < best_count:
            best_comps = pruned
            best_count = len(pruned)
    
    # 3. Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        searched = _local_search(best_comps, N, buffer_min, buffer_max, base_matrix, deadline)
        if len(searched) < best_count:
            best_comps = searched
            best_count = len(searched)
        elif len(searched) == best_count:
            # If same count, maybe keep the one that is "better" in some other metric?
            # Or just keep the first one.
            pass
    
    return best_comps
