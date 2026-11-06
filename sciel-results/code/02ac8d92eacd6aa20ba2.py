"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _batcher_merge(k):
    """
    Generate Batcher's merge network for two sorted lists of size k.
    Total wires: 2k. Returns list of (i, j) comparators.
    Uses the recursive odd-even merge definition.
    """
    if k == 1:
        return [(0, 1)]
    
    comparators = []
    # Batcher's merge: for two sorted sequences of length k
    # Step 1: Compare elements that are k apart (the "even" phase of merge)
    # Actually, Batcher's merge is defined as:
    #   M(k) = M(k/2) on odd positions + M(k/2) on even positions + cross compares
    # 
    # Standard Batcher merge for 2k elements from two sorted k-element lists:
    # 1. Split into two sub-merges of size k (interleaved)
    # 2. Cross-compare
    #
    # Let's use the standard recursive definition:
    # merge(k) for two sorted lists of length k (total 2k wires):
    #   if k == 1: [(0,1)]
    #   else:
    #     # The merge consists of:
    #     # 1. Apply merge(k/2) to the "odd" and "even" subsequences
    #     #    In terms of the 2k wires, this means:
    #     #    - Apply merge(k/2) to wires [0, 2, 4, ..., 2k-2] (even indices)
    #     #    - Apply merge(k/2) to wires [1, 3, 5, ..., 2k-1] (odd indices)
    #     # 2. Then compare (i, i+k) for i in 0..k-1
    #
    # Wait, that's not quite right. Let me think more carefully.
    #
    # Batcher's odd-even merge for two sorted lists A[0..k-1] and B[0..k-1]:
    # The combined list is C[0..2k-1] where C[i] = A[i] if i<k, B[i-k] if i>=k
    # But actually in the network, we interleave them.
    #
    # The standard Batcher OES merge network for 2k inputs:
    # It is defined recursively. For k=1: [(0,1)]
    # For k>1: 
    #   Let h = k//2 (assuming k is power of 2)
    #   1. Apply merge(h) to positions [0, 2, 4, ..., 2k-2] (the even-indexed wires)
    #      These are wires 0,2,4,...,2k-2, which form two sorted lists of size h
    #      (the even elements from A and B interleaved)
    #   2. Apply merge(h) to positions [1, 3, 5, ..., 2k-1] (the odd-indexed wires)
    #   3. Compare (i, i+k) for i = 0, 1, ..., k-1
    #
    # Actually, I need to be more careful. The standard definition is:
    # M(k) = [M(k/2) on even positions] + [M(k/2) on odd positions] + [(i, i+k) for i in range(k)]
    #
    # But the "even positions" in a 2k-element array are indices 0,2,4,...,2k-2
    # and "odd positions" are 1,3,5,...,2k-1.
    #
    # When we apply M(k/2) to the even positions, we're treating them as a k-element
    # problem. The mapping from the k-element sub-problem to the actual wire indices
    # is: sub-wire i -> actual wire 2*i.
    #
    # Similarly for odd positions: sub-wire i -> actual wire 2*i + 1.
    
    h = k // 2
    if h < 1:
        return [(0, 1)]
    
    # Apply merge(h) to even-indexed wires: 0, 2, 4, ..., 2k-2
    # The sub-problem has h elements (since we're merging two lists of size h
    # from the even positions), so we need merge(h) which operates on 2h = k wires.
    # Wait, no. merge(h) is for merging two sorted lists of size h, so it operates
    # on 2h wires. The even positions give us k wires total, which is 2*(k/2) = 2h.
    # So we apply merge(h) where the 2h wires are mapped to actual wires 0,2,4,...,2k-2.
    
    even_merge = _batcher_merge(h)
    for (a, b) in even_merge:
        comparators.append((2 * a, 2 * b))
    
    # Apply merge(h) to odd-indexed wires: 1, 3, 5, ..., 2k-1
    odd_merge = _batcher_merge(h)
    for (a, b) in odd_merge:
        comparators.append((2 * a + 1, 2 * b + 1))
    
    # Cross-compare: (i, i+k) for i in 0..k-1
    for i in range(k):
        comparators.append((i, i + k))
    
    return comparators

def _batcher_sort(n):
    """
    Generate Batcher's Odd-Even Mergesort network for n inputs (n must be power of 2).
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    half = n // 2
    left = _batcher_sort(half)
    right = _batcher_sort(half)
    
    # Shift right comparators by half
    right = [(i + half, j + half) for (i, j) in right]
    
    # Merge phase
    merge = _batcher_merge(half)
    
    return left + right + merge

def _verify_network(network):
    """Verify the network sorts all 2^16 Boolean inputs using zero-one principle."""
    if not network:
        return False
    for mask in range(1 << N):
        vec = [(mask >> i) & 1 for i in range(N)]
        for i, j in network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        for i in range(N - 1):
            if vec[i] > vec[i + 1]:
                return False
    return True

# Precompute the Batcher OES network for 16 wires
_BATCHER_16 = _batcher_sort(16)

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    return _BATCHER_16
# EVOLVE-BLOCK-END

