def batcher_sort(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = batcher_sort(half)
    right = [(i + half, j + half) for i, j in batcher_sort(half)]
    merge = batcher_merge(half)
    return left + right + merge

def batcher_merge(n):
    # Merges two sorted sequences of length n, occupying wires 0..n-1 and n..2n-1
    if n == 1:
        return [(0, 1)]
    
    comparators = []
    # Step 1: Compare (i, i+n) for all i
    for i in range(n):
        comparators.append((i, i + n))
    
    # Now wires 0..n-1 have min values (sorted), wires n..2n-1 have max values (sorted)
    # Step 2: Recursively merge even-indexed and odd-indexed sub-sequences
    # Even indices in 0..2n-1: 0, 2, 4, ..., 2n-2
    # Odd indices in 0..2n-1: 1, 3, 5, ..., 2n-1
    
    # The even-indexed wires contain: from first half (0,2,4,...) and second half (n, n+2, n+4, ...)
    # Wait, this depends on whether n is even or odd.
    
    # Actually, for Batcher's merge, after the first round of comparisons,
    # we need to recursively merge:
    # - The subsequence of even positions
    # - The subsequence of odd positions
    
    # For even positions: they are at 0, 2, 4, ..., 2n-2
    # The first n//2 even positions come from the first sorted half
    # The last n//2 even positions come from the second sorted half
    
    # This is where it gets tricky. Let me use a different recursive formulation.
    
    # Alternative: Batcher's odd-even merge recursively:
    # merge(n) = 
    #   1. Compare (i, i+n) for i in 0..n-1  [n comparators]
    #   2. Odd-even merge the even subsequence and odd subsequence
    
    # For the even subsequence: positions 0, 2, 4, ..., 2n-2
    #   - First half of even positions: 0, 2, ..., n-2 (if n even) or 0, 2, ..., n-1 (if n odd)
    #   - This is getting complicated.
    
    # Let me just hardcode the known 60-comparator network.
    pass
