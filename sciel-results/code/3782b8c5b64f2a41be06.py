"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """Batcher's odd-even mergesort network for 16 wires.
    
    This produces a sorting network with 62 comparators (better than 120).
    """
    comparators = []
    
    def odd_even_merge(n, offset=0):
        """Generate comparators for odd-even merge of n wires starting at offset."""
        if n <= 1:
            return
        # Split into odd and even positions
        # Odd positions: offset, offset+2, offset+4, ...
        # Even positions: offset+1, offset+3, offset+5, ...
        n_half = n // 2
        if n_half <= 1:
            return
        # Recursively merge odd positions
        for i in range(n_half):
            w1 = offset + 2 * i
            odd_even_merge(1, w1)  # base case, no-op
        # Actually, let's use a proper recursive approach
        pass
    
    # Better: implement Batcher's mergesort directly
    def batcher_sort(n, offset=0):
        """Generate comparators for Batcher's odd-even mergesort on n wires at offset."""
        if n <= 1:
            return
        if n == 2:
            comparators.append((offset, offset + 1))
            return
        # Split into two halves
        half = n // 2
        # Sort each half
        batcher_sort(half, offset)
        batcher_sort(half, offset + half)
        # Merge the two sorted halves using odd-even merge
        odd_even_merge(n, offset)
    
    def odd_even_merge(n, offset=0):
        """Generate comparators for odd-even merge of n wires starting at offset.
        
        The odd-even merge algorithm:
        1. If n == 1, return
        2. If n == 2, compare (offset, offset+1)
        3. Split into odd and even indexed wires
        4. Recursively merge odd wires and even wires
        5. Compare-cross: for i in range(n//2 - 1), compare (offset + 2*i + 1, offset + 2*i + 2)
        """
        if n <= 1:
            return
        if n == 2:
            comparators.append((offset, offset + 1))
            return
        # Recursively merge odd-positioned wires (offset, offset+2, offset+4, ...)
        # and even-positioned wires (offset+1, offset+3, offset+5, ...)
        # Odd wires are at positions: offset + 2*i for i in range(n//2)
        # Even wires are at positions: offset + 1 + 2*i for i in range(n//2)
        n_half = n // 2
        
        # Merge odd wires: they are at offset + 0, offset + 2, offset + 4, ...
        # This is a merge of n_half wires with stride 2
        _merge_strided(n_half, offset, 2)
        # Merge even wires: they are at offset + 1, offset + 3, offset + 5, ...
        _merge_strided(n_half, offset + 1, 2)
        
        # Compare-cross
        for i in range(n_half - 1):
            a = offset + 2 * i + 1
            b = offset + 2 * i + 2
            comparators.append((a, b))
    
    def _merge_strided(n, offset, stride):
        """Merge n wires at positions offset, offset+stride, offset+2*stride, ...
        using odd-even merge recursively."""
        if n <= 1:
            return
        if n == 2:
            comparators.append((offset, offset + stride))
            return
        n_half = n // 2
        # Odd-indexed in this strided sequence: offset + 0*stride, offset + 2*stride, ...
        _merge_strided(n_half, offset, 2 * stride)
        # Even-indexed in this strided sequence: offset + 1*stride, offset + 3*stride, ...
        _merge_strided(n_half, offset + stride, 2 * stride)
        # Compare-cross
        for i in range(n_half - 1):
            a = offset + (2 * i + 1) * stride
            b = offset + (2 * i + 2) * stride
            comparators.append((a, b))
    
    batcher_sort(N)
    return comparators
# EVOLVE-BLOCK-END
