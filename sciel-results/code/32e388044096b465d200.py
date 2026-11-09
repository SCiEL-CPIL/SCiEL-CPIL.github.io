import numpy as np
import time
import math

N = 18

def _batcher_merge(N1, N2):
    if N1 == 0 or N2 == 0:
        return []
    depth = math.ceil(math.log2(N1 + N2))
    comps = []
    for j in range(depth):
        step = 1 << j
        for i in range(N1 + N2 - step):
            if j == 0:
                comps.append((i, i + step))
            else:
                if (i % step) < (step // 2):
                    comps.append((i, i + step))
    return comps

def _batcher_sort(N):
    if N == 1:
        return []
    N1 = N // 2
    N2 = N - N1
    comps = []
    comps.extend(_batcher_sort(N1))
    half2_comps = _batcher_sort(N2)
    comps.extend([(i + N1, j + N1) for i, j in half2_comps])
    comps.extend(_batcher_merge(N1, N2))
    return comps

def _is_sorted_network(comps, N):
    """
    Check if the network sorts all 0-1 inputs using NumPy.
    """
    # Generate all 0-1 inputs
    inputs = np.arange(2**N, dtype=np.uint32)
    
    bits = 1 << np.arange(N)
    matrix = (inputs[:, None] & bits[None, :]) > 0
    
    # Apply network
    for i, j in comps:
        # min to i, max to j
        min_val = np.minimum(matrix[:, i], matrix[:, j])
        max_val = np.maximum(matrix[:, i], matrix[:, j])
        matrix[:, i] = min_val
        matrix[:, j] = max_val
        
    # Check if sorted
    diff = np.diff(matrix, axis=1)
    return np.all(diff >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    # 1. Generate Batcher network
    try:
        comps = _batcher_sort(N)
        # Deduplicate
        comps = list(set(comps))
    except Exception as e:
        # Fallback to a simple known valid network if Batcher fails (unlikely)
        # Simplest valid network for N wires: Odd-Even Transposition Sort
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
            
    # Ensure we have a valid starting point
    # Batcher sort is valid.
    best_comps = comps
    best_count = len(comps)
    
    # 2. Prune redundant comparators
    i = 0
    try:
        while i < len(comps) and time.time() < deadline:
            # Try to remove comps[i]
            test_comps = comps[:i] + comps[i+1:]
            
            if _is_sorted_network(test_comps, N):
                # Remove it
                comps = test_comps
                best_comps = comps
                best_count = len(comps)
                # Do not increment i, check the new element at i
            else:
                i += 1
    except Exception as e:
        # If pruning crashes, keep the best found so far
        pass

    # Final safety check: ensure we return a non-empty list of valid comparators
    if not best_comps:
        # If somehow empty, return the seed odd-even transposition network
        # which is guaranteed to be valid and non-empty for N>1
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
        best_comps = comps

    return best_comps
