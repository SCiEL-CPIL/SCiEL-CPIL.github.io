<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Odd-even transposition sorting network: 120 comparators for 16 wires.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def _batcher_oes(n):
    """Batcher's Odd-Even Merge Sort network for n wires (n power of 2).
    Returns list of (i, j) comparators."""
    if n == 1:
        return []
    k = n // 2
    # Recursively sort odd and even positions
    odd = _batcher_oes(k)
    even = _batcher_oes(k)
    # Map odd indices: wire 2*i+1, even: wire 2*i
    odd_mapped = [(2*i+1, 2*j+1) for i, j in odd]
    even_mapped = [(2*i, 2*j) for i, j in even]
    
    # Odd-even merge: merge two sorted halves of size k
    merge = _batcher_merge(k)
    
    return odd_mapped + even_mapped + merge

def _batcher_merge(k):
    """Batcher's odd-even merge of two sorted sequences of length k.
    The two halves are wires [0..k-1] and [k..2k-1].
    Returns list of (i, j) comparators."""
    if k == 1:
        return [(0, 1)]
    half = k // 2
    # Recursively merge first halves and second halves
    left = _batcher_merge(half)
    right = _batcher_merge(half)
    # Map: left operates on wires [0..half-1] and [half..k-1]
    # right operates on wires [k..k+half-1] and [k+half..2k-1]
    right_mapped = [(i + k, j + k) for i, j in right]
    
    # Cross comparisons: compare wire i with wire k-i for i in 0..half-1
    # Actually, the standard odd-even merge:
    # After merging first halves (wires 0..half-1 with half..k-1)
    # and second halves (wires k..k+half-1 with k+half..2k-1),
    # we need cross comparisons between the two merged results.
    # 
    # Standard Batcher OES merge for two sorted arrays A[0..k-1] and B[0..k-1]:
    # 1. Recursively merge A[0..half-1] with B[0..half-1] -> sorted in wires 0..half-1
    # 2. Recursively merge A[half..k-1] with B[half..k-1] -> sorted in wires half..k-1
    #    Wait, that's not quite right. Let me reconsider.
    #
    # Actually for OES merge of two sorted sequences of size k:
    # Input: first half wires 0..k-1 sorted, second half wires k..2k-1 sorted
    # Step 1: Recursively merge wires [0..half-1] with [k..k+half-1]
    #         and wires [half..k-1] with [k+half..2k-1]
    # Step 2: Compare-exchange wire i with wire 2*half-1-i for i in 0..half-1
    #         i.e., compare wire i with wire k-1-i... 
    
    # Let me use the standard formulation:
    # For merging two sorted lists of size k (in wires 0..k-1 and k..2k-1):
    # if k == 1: compare (0,1)
    # else:
    #   h = k//2
    #   merge(0..h-1 with h..k-1)  -- first halves
    #   merge(k..k+h-1 with k+h..2k-1)  -- second halves  
    #   for i in 0..h-1: compare(i, 2*h-1-i)  -- cross
    
    # Wait, I need to be more careful. Let me use a cleaner formulation.
    # The two input sorted sequences occupy wires [0, k) and [k, 2k).
    # After recursive merges and cross comparisons, wires [0, 2k) are sorted.
    
    # Standard Batcher OES merge:
    # merge(A[0..k-1], B[0..k-1]) where A is in wires 0..k-1, B in wires k..2k-1
    # if k=1: compare(0,1)
    # if k>1: h=k//2
    #   merge(A[0..h-1], B[0..h-1])  -> wires 0..h-1 get min of each pair
    #   merge(A[h..k-1], B[h..k-1])  -> wires h..k-1... 
    #   Hmm, this is getting confusing. Let me just hardcode for n=16.
    
    # Actually, let me just build it directly for n=16 using the known structure.
    return []  # placeholder, will be replaced

def construct():
    """Batcher's Odd-Even Merge Sort network for 16 wires: 63 comparators."""
    # Known 63-comparator Batcher OES network for n=16
    # Built from recursive structure:
    # n=16: sort evens (wires 0,2,4,6,8,10,12,14) and odds (wires 1,3,5,7,9,11,13,15)
    #   each of size 8, then merge
    # n=8: sort evens (size 4) and odds (size 4), then merge
    # n=4: sort evens (size 2) and odds (size 2), then merge
    # n=2: compare(0,1)
    # n=1: nothing
    
    # Let's build it properly with a recursive function
    def oes(n):
        """Return comparators for Batcher OES on wires 0..n-1 (n power of 2)."""
        if n == 1:
            return []
        k = n // 2
        # Sort even-indexed wires: 0, 2, 4, ..., 2k-2
        # Sort odd-indexed wires: 1, 3, 5, ..., 2k-1
        even_sort = oes(k)  # operates on positions 0..k-1, maps to wires 0,2,4,...
        odd_sort = oes(k)   # operates on positions 0..k-1, maps to wires 1,3,5,...
        
        even_mapped = [(2*i, 2*j) for i, j in even_sort]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_sort]
        
        # Now merge the two sorted halves (size k each)
        merge = merge_network(k)
        
        return even_mapped + odd_mapped + merge
    
    def merge_network(k):
        """Odd-even merge of two sorted sequences of size k.
        First sequence in wires 0..k-1, second in wires k..2k-1.
        Returns comparators to produce sorted output in wires 0..2k-1."""
        if k == 1:
            return [(0, 1)]
        h = k // 2
        # Recursively merge first halves: wires 0..h-1 with wires k..k+h-1
        # This is a merge of size h, but the wires are offset.
        # We need a merge that works on arbitrary wire positions.
        # Let's use a helper that takes offsets.
        
        left = merge_network_offset(h, 0, k)      # merge wires [0..h-1] with [k..k+h-1]
        right = merge_network_offset(h, h, k+h)   # merge wires [h..k-1] with [k+h..2k-1]
        
        # Cross comparisons: for i in 0..h-1, compare wire i with wire 2*h-1-i
        # Wait: after the recursive merges, wires 0..h-1 contain the smaller elements
        # from the first merge, and wires h..k-1 contain the larger elements from the first merge.
        # Similarly wires k..k+h-1 and k+h..2k-1.
        # The cross comparisons connect wire i (from first half's result) with wire 2*h-1-i.
        # Actually: compare wire i with wire (2*h - 1 - i) for i = 0, 1, ..., h-1.
        # But wait, 2*h = k, so compare wire i with wire k-1-i for i=0..h-1.
        # Hmm, that doesn't seem right either. Let me think again.
        
        # Standard Batcher OES merge for two sorted sequences of length k:
        # Input: X[0..k-1] sorted in wires 0..k-1, Y[0..k-1] sorted in wires k..2k-1
        # h = k//2
        # Step 1: Merge X[0..h-1] with Y[0..h-1] -> result in wires 0..h-1
        # Step 2: Merge X[h..k-1] with Y[h..k-1] -> result in wires h..k-1  
        # Step 3: For i = 0 to h-1: compare-exchange wire i with wire (2*h-1-i)
        #         i.e., wire i with wire (k-1-i)
        # Wait no. Let me look at this more carefully.
        
        # The correct Batcher OES merge:
        # merge(A, B) where A = [a0, a1, ..., a_{k-1}] sorted, B = [b0, b1, ..., b_{k-1}] sorted
        # If k = 1: compare a0 with b0
        # If k > 1: h = k//2
        #   merge(A[0..h-1], B[0..h-1])  -> produces sorted sequence of size h in wires 0..h-1
        #   merge(A[h..k-1], B[h..k-1])  -> produces sorted sequence of size h in wires h..k-1
        #   for i = 0 to h-1: compare wire i with wire (2*h-1-i)
        #   where 2*h = k, so compare wire i with wire (k-1-i)
        
        # But wait, after step 1, wires 0..h-1 contain the h smallest elements from A[0..h-1] and B[0..h-1].
        # After step 2, wires h..k-1 contain the h largest elements from A[h..k-1] and B[h..k-1].
        # But we also need wires k..2k-1 to be involved. I think I'm confusing myself.
        
        # Let me use a different approach: define merge on specific wire ranges.
        return left + right + cross_comparisons(k, h)
    
    def cross_comparisons(k, h):
        """Cross comparisons for OES merge of size k (h=k//2).
        After recursive merges, compare wire i with wire (2*h-1-i) for i=0..h-1."""
        return [(i, 2*h-1-i) for i in range(h)]
    
    def merge_network_offset(h, offset1, offset2):
        """Merge two sorted sequences of size h in wires [offset1..offset1+h-1] and [offset2..offset2+h-1]."""
        if h == 1:
            return [(offset1, offset2)]
        hh = h // 2
        left = merge_network_offset(hh, offset1, offset2)
        right = merge_network_offset(hh, offset1+hh, offset2+hh)
        cross = [(offset1+i, offset1+2*hh-1-i) for i in range(hh)]
        return left + right + cross
    
    # Now build the full network
    def oes_full(n):
        """Batcher OES for n wires."""
        if n == 1:
            return []
        k = n // 2
        even_sort = oes_full(k)
        odd_sort = oes_full(k)
        even_mapped = [(2*i, 2*j) for i, j in even_sort]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_sort]
        merge = merge_network_offset(k, 0, k)
        return even_mapped + odd_mapped + merge
    
    network = oes_full(16)
    return network
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, let me reconsider the merge structure. The Batcher OES merge is subtle. Let me trace through it more carefully and verify with a known-correct implementation.

