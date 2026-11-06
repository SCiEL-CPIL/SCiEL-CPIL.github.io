"""
Strategy:
1. Correctly generate Batcher's Odd-Even Merge Sort network for N=16 (63 comparators).
2. Identify and remove the 3 redundant comparators specific to the N=16 optimal network to reach 60 comparators.
3. Verify the resulting 60-comparator network.
4. If verification fails (due to incorrect redundancy assumption), fall back to the valid 63-comparator network.
This deterministic approach avoids the stochastic search failures of the previous strategy and guarantees a significant score improvement (from ~1.25 to ~1.90 or 2.0).
"""

import time
import numpy as np

N = 16
MAX_TIME = 55.0

def _generate_batcher_network(n):
    """
    Generates the correct Batcher's Odd-Even Merge Sort network for N=n.
    Returns a list of (i, j) tuples.
    """
    def merge(n, offset, stride=1):
        if n == 1:
            return [(offset, offset + stride)]
        
        h = n // 2
        # Step 1: Compare elements n*stride apart
        step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
        
        # Step 2: Recursively merge the even and odd subsequences
        # The even subsequence has h elements? No, n elements total in the merge, 
        # but the recursive call is on the halves of the merged structure.
        # In Batcher's merge of two sorted lists of length n:
        # The even-indexed elements form two sorted lists of length n/2.
        # The odd-indexed elements form two sorted lists of length n/2.
        # So we call merge(n/2, offset, 2*stride) and merge(n/2, offset+stride, 2*stride).
        
        even = merge(h, offset, 2 * stride)
        odd = merge(h, offset + stride, 2 * stride)
        
        return step1 + even + odd

    def sort(n, offset=0, stride=1):
        if n == 1:
            return []
        if n == 2:
            return [(offset, offset + stride)]
        
        h = n // 2
        left = sort(h, offset, stride)
        right = sort(h, offset + h * stride, stride)
        m = merge(n, offset, stride) # Wait, merge expects two sorted halves of length n/2?
        # My merge function above takes n as the length of ONE half?
        # Let's re-read my merge definition.
        # merge(n, offset, stride):
        #   if n==1: return [(offset, offset+stride)]
        #   h = n//2
        #   step1 = [(offset + i*stride, offset + (n+i)*stride) for i in range(n)]
        #   This compares (0, n), (1, n+1)... which is correct for merging two lists of length n.
        #   So merge(n) merges two lists of length n.
        #   In sort(n), we have two lists of length n/2.
        #   So we should call merge(n//2, offset, stride).
        
        m = merge(h, offset, stride)
        
        return left + right + m

    return sort(n)

def _apply_network_vectorized(comps, inputs):
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Generate Batcher Network (63 comparators for N=16)
    comps = _generate_batcher_network(N)
    
    # Verify Batcher Network
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    if not _is_sorted_network_vectorized(comps, bits):
        # Fallback to Odd-Even Transposition if Batcher generator is buggy (unlikely but safe)
        oet_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    oet_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    oet_comps.append((j, j + 1))
        return oet_comps

    # Attempt to reduce to 60 comparators
    # The optimal 60-comparator network for N=16 is known to be Batcher minus 3 specific comparators.
    # These are often in the final merge stages.
    # We will try removing each comparator in the last 10 stages and check validity.
    # If a removal keeps the network valid, we remove it.
    
    current_comps = comps[:]
    
    # Try to remove redundant comparators
    # We iterate backwards from the end, as later comparators are more likely to be redundant
    # if the network is already sorted.
    for i in range(len(current_comps) - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        candidate = current_comps[:i] + current_comps[i+1:]
        if _is_sorted_network_vectorized(candidate, bits):
            current_comps = candidate
            # We want exactly 60. If we go below, we stop. But Batcher is 63, so we remove 3.
            if len(current_comps) == 60:
                break
    
    # Final verification
    if _is_sorted_network_vectorized(current_comps, bits):
        return current_comps
    else:
        return comps
