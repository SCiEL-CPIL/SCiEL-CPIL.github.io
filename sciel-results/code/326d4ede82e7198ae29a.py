# EVOLVE-BLOCK-START

def _apply_comparator(state, i, j):
    """Apply comparator (i, j) to a bitmask state.
    state is a tuple of 16 bits (0/1).
    Returns new tuple with min at i, max at j.
    """
    if state[i] > state[j]:
        lst = list(state)
        lst[i], lst[j] = lst[j], lst[i]
        return tuple(lst)
    return state

def _is_sorted(state):
    """Check if state is sorted (non-decreasing)."""
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _verify_network(comparators):
    """Verify that a network sorts all 2^16 Boolean inputs using zero-one principle."""
    # Fast verification using bitmask
    def apply_comp_mask(m, i, j):
        bi = (m >> i) & 1
        bj = (m >> j) & 1
        if bi > bj:
            m ^= (1 << i) | (1 << j)
        return m
    
    # Precompute popcount for all 16-bit masks to speed up check
    # bin(m).count('1') is okay, but let's try to be efficient.
    
    for mask in range(1 << N):
        m = mask
        for (i, j) in comparators:
            m = apply_comp_mask(m, i, j)
        # Check if m is sorted: all 0s followed by all 1s
        # A sorted 16-bit mask has the form: 000...0111...1
        # So if k ones, m should be (1 << k) - 1
        ones = bin(m).count('1')
        expected = (1 << ones) - 1
        if m != expected:
            return False
    return True

def _depth_of_network(comparators):
    """Compute parallel depth of a comparator network."""
    if not comparators:
        return 0
    depths = [0] * N
    max_depth = 0
    for (i, j) in comparators:
        d = max(depths[i], depths[j]) + 1
        depths[i] = d
        depths[j] = d
        if d > max_depth:
            max_depth = d
    return max_depth

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network (120 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_merge_sort(n):
    """Generate Batcher's Odd-Even Merge Sort network for n wires."""
    if n == 1:
        return []
    
    def odd_even_merge(n, offset):
        """Generate comparators to merge two sorted halves of length n/2 starting at offset.
        The two halves are already sorted.
        Returns list of (i, j) with i < j.
        """
        if n == 2:
            return [(offset, offset + 1)]
        
        half = n // 2
        comps = []
        # Step 1: Compare elements at distance half
        for i in range(half):
            comps.append((offset + i, offset + i + half))
        
        # Step 2: Recursively merge the even-indexed wires and odd-indexed wires
        # The even-indexed wires of the first half and second half form a bitonic sequence?
        # Actually, after step 1, the even wires (offset, offset+2, ...) form a bitonic sequence
        # and the odd wires (offset+1, offset+3, ...) form a bitonic sequence.
        # We need to sort them.
        
        # Recursive call on even wires: indices offset, offset+2, ..., offset+n-2
        # These are not contiguous in the original array, but we can map them to a virtual array?
        # No, we must use original indices.
        # The comparator network for the even wires operates on indices:
        # offset, offset+2, offset+4...
        # We can generate the network for size n/2 and then map indices.
        
        def map_indices(net, start, step):
            """Map a network generated for contiguous indices 0..m-1 to indices start, start+step, ..."""
            mapped = []
            for (i, j) in net:
                new_i = start + i * step
                new_j = start + j * step
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                # Ensure i < j for the interface
                if new_i < new_j:
                    mapped.append((new_i, new_j))
                else:
                    # This shouldn't happen if step > 0 and i < j
                    mapped.append((new_i, new_j))
            return mapped

        # Generate merge network for size half (which is n/2)
        # Wait, the recursive merge is on the *sequence* of even wires.
        # The length of the even wire sequence is half (n/2).
        # So we call odd_even_merge(half, 0) to get the pattern, then map it.
        
        even_net = odd_even_merge(half, 0)
        # Map even_net to indices offset, offset+2, ...
        mapped_even = []
        for (i, j) in even_net:
            new_i = offset + i * 2
            new_j = offset + j * 2
            if new_i < new_j:
                mapped_even.append((new_i, new_j))
            else:
                mapped_even.append((new_j, new_i)) # Should be i<j always if i<j in input
        
        odd_net = odd_even_merge(half, 0)
        # Map odd_net to indices offset+1, offset+3, ...
        mapped_odd = []
        for (i, j) in odd_net:
            new_i = offset + 1 + i * 2
            new_j = offset + 1 + j * 2
            if new_i < new_j:
                mapped_odd.append((new_i, new_j))
            else:
                mapped_odd.append((new_j, new_i))
                
        comps += mapped_even
        comps += mapped_odd
        return comps

    def sort(n, offset):
        if n == 1:
            return []
        half = n // 2
        comps = []
        # Sort first half
        comps += sort(half, offset)
        # Sort second half
        comps += sort(half, offset + half)
        # Merge
        comps += odd_even_merge(n, offset)
        return comps

    return sort(n, 0)

def construct():
    """Construct a sorting network for 16 wires."""
    # Try Batcher's Odd-Even Merge Sort (80 comparators)
    net = _batcher_odd_merge_sort(N)
    
    # Verify it
    if _verify_network(net):
        return net
    
    # Fallback to Odd-Even Transposition (120 comparators)
    net_oe = _odd_even_transposition()
    if _verify_network(net_oe):
        return net_oe
    
    # Last resort
    return net_oe

# EVOLVE-BLOCK-END
