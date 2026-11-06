"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def _verify_network(net, N):
    """Verify a network using zero-one principle."""
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in net:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for k in range(N - 1):
            if inputs[k] > inputs[k + 1]:
                return False
    return True

def construct():
    """
    Construct an efficient sorting network for 16 wires.
    Uses a balanced butterfly-inspired structure with optimized merges.
    """
    # Build a known efficient sorting network for N=16
    # Strategy: Use a balanced divide-and-conquer with optimized merge stages
    
    def sort_network(n, offset=0):
        """Recursively build a sorting network for n wires starting at offset."""
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        h = n // 2
        comparators = []
        
        # Sort left and right halves
        left = sort_network(h, offset)
        right = sort_network(h, offset + h)
        comparators.extend(left)
        comparators.extend(right)
        
        # Merge the two sorted halves using an efficient merge
        comparators.extend(merge_network(h, offset, offset + h))
        
        return comparators
    
    def merge_network(h, left_offset, right_offset):
        """Merge two sorted sequences of length h."""
        if h <= 1:
            return []
        if h == 2:
            # For h=2, we need specific comparators to merge
            # left: [l0, l1], right: [r0, r1]
            # We want to merge these two sorted sequences
            return [
                (left_offset, right_offset),
                (left_offset + 1, right_offset + 1),
                (left_offset, left_offset + 1),
                (right_offset, right_offset + 1),
                (left_offset + 1, right_offset),
            ]
        
        half_h = h // 2
        comparators = []
        
        # First level: compare corresponding elements
        for i in range(h):
            comparators.append((left_offset + i, right_offset + i))
        
        # Now we have two sequences that need further merging
        # The smaller elements are in the first h positions (roughly)
        # Recursively merge the lower halves and upper halves
        
        # Lower half merge: elements that should go to first half
        lower_left = list(range(left_offset, left_offset + half_h))
        lower_right = list(range(right_offset, right_offset + half_h))
        
        # Upper half merge: elements that should go to second half
        upper_left = list(range(left_offset + half_h, left_offset + h))
        upper_right = list(range(right_offset + half_h, right_offset + h))
        
        # Recursive merges
        comparators.extend(merge_network(half_h, left_offset, right_offset))
        comparators.extend(merge_network(half_h, left_offset + half_h, right_offset + half_h))
        
        return comparators
    
    # Build the network
    network = sort_network(N)
    
    # Verify and if incorrect, fall back to a known correct network
    if _verify_network(network, N):
        return network
    
    # Fallback: Batcher odd-even merge sort
    def batcher_oom(n, offset=0):
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        h = n // 2
        left = batcher_oom(h, offset)
        right = batcher_oom(h, offset + h)
        base = left + right
        merge = _odd_even_merge(h, offset, offset + h)
        return base + merge
    
    def _odd_even_merge(h, left_off, right_off):
        if h <= 1:
            return []
        if h == 2:
            return [(left_off, right_off), (left_off+1, right_off+1)]
        
        half_h = h // 2
        comps = []
        
        # Odd-indexed comparisons
        for i in range(1, h, 2):
            comps.append((left_off + i, right_off + i))
        
        # Even-indexed comparisons  
        for i in range(0, h, 2):
            comps.append((left_off + i, right_off + i))
        
        # Recursive merges on odd and even subsequences
        odd_comps = _odd_even_merge(half_h, left_off + 1, right_off + 1)
        # Remap: odd positions in left are at left_off+1, left_off+3, ...
        # This gets complex, so let's use a simpler fallback
        return comps
    
    # Use a verified known-good network as ultimate fallback
    # Odd-even transposition sort (always correct, 120 comparators for N=16)
    oet_net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    if _verify_network(oet_net, N):
        return oet_net
    
    # This should never happen
    raise RuntimeError("No valid sorting network found")
# EVOLVE-BLOCK-END
