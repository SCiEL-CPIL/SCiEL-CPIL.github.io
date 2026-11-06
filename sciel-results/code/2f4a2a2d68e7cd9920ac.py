"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a Bitonic Sorting Network for N=16 wires.
    Returns a list of (i, j) comparators.
    """
    def bitonic_sort(n, offset=0):
        """
        Recursively generates comparators for Bitonic Sort on wires [offset, offset+n).
        """
        if n <= 1:
            return []
        
        half = n // 2
        
        # 1. Sort the first half
        left = bitonic_sort(half, offset)
        # 2. Sort the second half
        right = bitonic_sort(half, offset + half)
        
        # 3. Merge the two halves
        merged = bitonic_merge(n, offset)
        
        return left + right + merged

    def bitonic_merge(n, offset):
        """
        Merges two sorted sequences of length n/2 into one sorted sequence of length n.
        Assumes the input is a bitonic sequence (first half ascending, second half descending 
        is NOT the assumption here; Bitonic Sort's merge step handles the structure 
        resulting from the recursive sort).
        
        Actually, the standard Bitonic Sort merge step:
        1. Compare (i, i + n/2) for i in 0..n/2-1.
        2. Recursively merge the first half (ascending) and second half (ascending).
        """
        if n <= 1:
            return []
        
        half = n // 2
        comparators = []
        
        # Step 1: Compare elements separated by n/2
        for i in range(half):
            comparators.append((offset + i, offset + i + half))
        
        # Step 2: Recursively merge the two halves
        # The first half (offset .. offset+half-1) and second half (offset+half .. offset+n-1)
        # are now bitonic? No, they are sorted in opposite directions?
        # In Bitonic Sort, after the comparisons in step 1, the first half is a bitonic sequence 
        # and the second half is a bitonic sequence.
        # We recursively sort them both in ascending order.
        
        left_merge = bitonic_merge(half, offset)
        right_merge = bitonic_merge(half, offset + half)
        
        comparators.extend(left_merge)
        comparators.extend(right_merge)
        
        return comparators

    return bitonic_sort(N)
# EVOLVE-BLOCK-END
