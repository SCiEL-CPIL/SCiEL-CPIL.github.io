"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def _verify_and_optimize(nets, N):
    """Verify a list of networks and return the best (fewest comparators) one."""
    best_net = None
    best_count = float('inf')
    for net in nets:
        if len(net) >= best_count:
            continue
        # Verify correctness using zero-one principle
        correct = True
        for mask in range(1 << N):
            inputs = [(mask >> i) & 1 for i in range(N)]
            for (i, j) in net:
                if inputs[i] > inputs[j]:
                    inputs[i], inputs[j] = inputs[j], inputs[i]
            for k in range(N - 1):
                if inputs[k] > inputs[k + 1]:
                    correct = False
                    break
            if not correct:
                break
        if correct:
            best_count = len(net)
            best_net = net
    return best_net

def construct():
    # Batcher odd-even-merge sort network for N=16
    def batcher_oom(n):
        """Generate Batcher's odd-even-merge sort network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        h = n // 2
        if n % 2 == 0:
            left = batcher_oom(h)
            right = batcher_oom(h)
            right = [(i + h, j + h) for (i, j) in right]
            base = left + right
            # Odd-even merge
            merge = _odd_even_merge(n)
            return base + merge
        else:
            # For odd n, use a different approach
            # Actually, Batcher OOM works for n = 2^k, so let's handle that
            # For N=16, this is fine
            raise ValueError("Only even n supported")
    
    def _odd_even_merge(n):
        """Generate the odd-even merge comparators for n wires (after recursive sorts)."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        h = n // 2
        # Odd merge: compare (1, h+1), (3, h+3), ...
        # Even merge: compare (0, h), (2, h+2), ...
        # Then recursively merge odd and even subsequences
        
        comparators = []
        
        # Odd merge: positions 1, 3, 5, ... vs h+1, h+3, h+5, ...
        # In 0-indexed: odd positions are 0, 2, 4, ... (0-based index)
        # Wait, let me be more careful.
        # Wires 0..n-1. Odd merge: compare wire i with wire i+h for i odd (1, 3, ...)
        # In 0-based: compare (i, i+h) for i = 1, 3, 5, ..., n-2
        for i in range(1, h, 2):
            comparators.append((i, i + h))
        
        # Even merge: compare wire i with wire i+h for i even (0, 2, 4, ...)
        for i in range(0, h, 2):
            comparators.append((i, i + h))
        
        # Now recursively merge the odd and even subsequences
        # Odd subsequence: wires 1, 3, 5, ..., n-1 -> remap to 0, 1, 2, ..., h-1
        # Even subsequence: wires 0, 2, 4, ..., n-2 -> remap to 0, 1, 2, ..., h-1
        odd_merge = _odd_even_merge(h)
        for (a, b) in odd_merge:
            # Map a, b from [0, h-1] to odd positions [1, 3, ..., 2h-1]
            comparators.append((2 * a + 1, 2 * b + 1))
        
        even_merge = _odd_even_merge(h)
        for (a, b) in even_merge:
            # Map a, b from [0, h-1] to even positions [0, 2, ..., 2h-2]
            comparators.append((2 * a, 2 * b))
        
        return comparators
    
    # Generate Batcher's OOM network for N=16
    batcher_net = batcher_oom(N)
    
    # Also try some other known good networks
    # Bitonic sort network for N=16
    def bitonic_sort(n):
        """Generate bitonic sort network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        h = n // 2
        comparators = []
        
        # Recursively sort first half ascending, second half descending
        left = bitonic_sort(h)
        right = bitonic_sort(h)
        right = [(i + h, j + h) for (i, j) in right]
        # For the right half, we want it descending, so we reverse the comparison direction
        # Actually, for bitonic sort, we sort one half ascending and one half descending
        # The comparators in the right half should be reversed to make it descending
        # But the comparator (i, j) always puts min at i, max at j
        # To make the right half descending, we can just reverse the order of comparisons
        # Actually, it's simpler to just use the same comparators but the bitonic merge will handle it
        
        comparators = left + right
        
        # Bitonic merge
        merge = _bitonic_merge(n)
        comparators = comparators + merge
        
        return comparators
    
    def _bitonic_merge(n):
        """Generate bitonic merge comparators for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        h = n // 2
        comparators = []
        
        # Compare (i, i+h) for i in range(n/2)
        for i in range(h):
            comparators.append((i, i + h))
        
        # Recursively merge first half and second half
        left_merge = _bitonic_merge(h)
        for (a, b) in left_merge:
            comparators.append((a, b))
        
        right_merge = _bitonic_merge(h)
        for (a, b) in right_merge:
            comparators.append((a + h, b + h))
        
        return comparators
    
    bitonic_net = bitonic_sort(N)
    
    # Odd-even transposition (the original)
    oet_net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify all and pick the best
    best_net = _verify_and_optimize([batcher_net, bitonic_net, oet_net], N)
    
    if best_net is None:
        # Fallback to OET
        best_net = oet_net
    
    return best_net
# EVOLVE-BLOCK-END
