import math
import numpy as np
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    n = 30
    
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Suffix sums R[i] = sum(W[i-1] ... W[n-1])
    R = [0] * (n + 2)
    for i in range(n, 0, -1):
        R[i] = R[i + 1] + W[i - 1]
    
    # Split into two halves: 1-15 and 16-30
    mid = 15
    
    # Beam search to generate pruned sums for each half
    # Keep only top-K sums closest to 0 at each step
    K = 5000  # Beam width
    
    def beam_generate(weights, K, deadline):
        """Generate pruned set of sums using beam search."""
        sums = [(0, [0] * len(weights))]  # (sum_value, coeffs)
        for i, w in enumerate(weights):
            new_sums = []
            for s, coeffs in sums:
                for c in [-1, 0, 1]:
                    new_s = s + c * w
                    new_coeffs = coeffs[:]
                    new_coeffs[i] = c
                    new_sums.append((new_s, new_coeffs))
            
            # Sort by absolute value and keep top K
            new_sums.sort(key=lambda x: abs(x[0]))
            sums = new_sums[:K]
            
            if time.time() > deadline:
                break
        
        return sums
    
    # Generate beam sums for first half (indices 0 to mid-1)
    weights_a = W[:mid]
    beam_a = beam_generate(weights_a, K, deadline)
    
    if time.time() > deadline:
        # Fallback to simple greedy
        return _greedy_fallback(W, n)
    
    # Generate beam sums for second half (indices mid to n-1)
    weights_b = W[mid:]
    beam_b = beam_generate(weights_b, K, deadline)
    
    if time.time() > deadline:
        return _greedy_fallback(W, n)
    
    # Extract just the sum values for efficient matching
    sums_a_vals = [s for s, _ in beam_a]
    sums_b_vals = [s for s, _ in beam_b]
    
    # Sort B for binary search
    sorted_b = sorted(enumerate(sums_b_vals), key=lambda x: x[1])
    arr_b_vals = np.array([x[1] for x in sorted_b])
    arr_b_idx = np.array([x[0] for x in sorted_b])
    
    best_val = float('inf')
    best_a_sum = None
    best_b_idx = None
    best_a_coeffs = None
    best_b_coeffs = None
    
    # Also track greedy solution as initial best
    coeffs_greedy = _greedy(W, n)
    total_greedy = sum(c * W[i] for i, c in enumerate(coeffs_greedy))
    if total_greedy != 0:
        best_val = abs(total_greedy)
        best_a_coeffs = coeffs_greedy[:mid]
        best_b_coeffs = coeffs_greedy[mid:]
    
    # Precompute max abs sum for B for pruning
    max_b_abs = max((abs(s) for s in sums_b_vals), default=0)
    
    # Meet in the middle: for each sum in A, find best matching sum in B
    for a_sum, a_coeffs in beam_a:
        if time.time() > deadline:
            break
        
        # Pruning
        if best_val < float('inf'):
            if abs(a_sum) - max_b_abs >= best_val:
                continue
        
        # Find b closest to -a_sum
        target = -a_sum
        idx = np.searchsorted(arr_b_vals, target)
        
        candidates = []
        if idx < len(arr_b_vals):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        
        for ci in candidates:
            b_val = arr_b_vals[ci]
            b_idx = arr_b_idx[ci]
            val = a_sum + b_val
            if val != 0:
                abs_val = abs(val)
                if abs_val < best_val:
                    best_val = abs_val
                    best_a_sum = a_sum
                    best_b_idx = b_idx
                    best_a_coeffs = a_coeffs
                    best_b_coeffs = beam_b[b_idx][1]
    
    # Reconstruct final coefficients
    if best_a_coeffs is not None and best_b_coeffs is not None:
        final_coeffs = best_a_coeffs + best_b_coeffs
        total = sum(c * W[i] for i, c in enumerate(final_coeffs))
        if total != 0:
            # Local search refinement
            final_coeffs = _local_search(final_coeffs, W, n, deadline)
            return final_coeffs
    
    # Fallback
    return _greedy_fallback(W, n)

def _greedy(W, n):
    """Simple greedy: pick coefficient to minimize |running sum| at each step."""
    coeffs = [0] * n
    curr_sum = 0
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_val_local = curr_sum
        for c in [-1, 0, 1]:
            val = curr_sum + c * w
            if abs(val) < abs(best_val_local):
                best_val_local = val
                best_c = c
        coeffs[i - 1] = best_c
        curr_sum = best_val_local
    return coeffs

def _greedy_fallback(W, n):
    """Ensure non-zero result from greedy."""
    coeffs = _greedy(W, n)
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
    return coeffs

def _local_search(coeffs, W, n, deadline):
    """Try flipping individual coefficients to reduce |sum|."""
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
        total = sum(c * W[i] for i, c in enumerate(coeffs))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for c_new in [-1, 0, 1]:
                if c_new == coeffs[i]:
                    continue
                new_total = total + (c_new - coeffs[i]) * W[i]
                if new_total != 0 and abs(new_total) < abs(total):
                    coeffs[i] = c_new
                    total = new_total
                    improved = True
                    break
    return coeffs
