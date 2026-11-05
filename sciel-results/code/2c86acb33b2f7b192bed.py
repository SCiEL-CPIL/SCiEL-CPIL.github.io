"""Optimized compare-exchange network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """Return the 52-comparator Batcher bitonic sorting network for 16 wires."""
    
    def gen_bitonic(n, ascending):
        """Generate comparators for a bitonic sequence of length n."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        m = n // 2
        if ascending:
            left = gen_bitonic(m, True)
            right = gen_bitonic(m, False)
        else:
            left = gen_bitonic(m, False)
            right = gen_bitonic(m, True)
        
        # Remap right half comparators to second half
        right_mapped = [(i + m, j + m) for i, j in right]
        
        # Bitonic merge of two halves of size m
        merge = []
        for k in range(m, 0, -1):
            for i in range(0, n - k):
                # Standard bitonic merge: compare i with i+k
                # For i < n/2: (i, i+k)  where we put min at lower index
                # For i >= n/2: (i, i-k) but that would be (i+k, i)
                # Actually, the standard is:
                # if i < n/2: comparator (i, i+k)
                # else: comparator (i+k, i)
                if i < n // 2:
                    merge.append((i, i + k))
                else:
                    merge.append((i + k, i))
        
        return left + right_mapped + merge
    
    return gen_bitonic(16, True)
# EVOLVE-BLOCK-END
