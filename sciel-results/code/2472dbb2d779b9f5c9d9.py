import time
import random
from functools import lru_cache

N = 18
TIME_LIMIT = 80  # seconds, leave margin for the 90s wall-clock limit


def _is_sorted_network(net):
    """Verify network sorts all 2^N Boolean inputs using 0-1 principle."""
    for mask in range(1 << N):
        wires = [(mask >> i) & 1 for i in range(N)]
        for a, b in net:
            if wires[a] > wires[b]:
                wires[a], wires[b] = wires[b], wires[a]
        if any(wires[i] > wires[i + 1] for i in range(N - 1)):
            return False
    return True


def _network_count(net):
    return len(net)


def _depth(net):
    """Compute parallel depth of a comparator network."""
    depth = [0] * N
    for a, b in net:
        d = max(depth[a], depth[b]) + 1
        depth[a] = d
        depth[b] = d
    return max(depth) if depth else 0


def _make_odd_even_transposition(n):
    """Standard odd-even transposition sorting network."""
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net


def _make_bitonic_merge(n):
    """
    Generate a bitonic merge network for n wires (n must be power of 2).
    Merges a bitonic sequence (first half ascending, second half descending)
    into a fully sorted sequence.
    """
    net = []
    if n <= 1:
        return net
    half = n // 2
    # Recursive bitonic merge
    # Compare wires that are 'half' apart
    for i in range(half):
        net.append((i, i + half))
    # Recursively sort the two halves
    left = _make_bitonic_merge(half)
    right = _make_bitonic_merge(half)
    # Offset right half by 'half'
    for a, b in left:
        net.append((a, b))
    for a, b in right:
        net.append((a + half, b + half))
    return net


def _make_bitonic_sort(n):
    """
    Generate a bitonic sorting network for n wires (n must be power of 2).
    """
    if n <= 1:
        return []
    
    def bitonic_sort_recursive(k):
        """Returns comparators to sort k wires (power of 2)."""
        if k == 1:
            return []
        half = k // 2
        # First, create bitonic sequences
        left = bitonic_sort_recursive(half)
        right = bitonic_sort_recursive(half)
        # Apply to right half with offset
        net = []
        for a, b in left:
            net.append((a, b))
        for a, b in right:
            net.append((a + half, b + half))
        # Now merge the bitonic sequences
        merge = _make_bitonic_merge(k)
        net.extend(merge)
        return net
    
    return bitonic_sort_recursive(n)


def _pad_to_n(net, n):
    """
    If net is for a smaller power of 2, we can't simply pad.
    Instead, this is a placeholder - we need a different approach for non-power-of-2.
    """
    return net


def _construct_small_n(n):
    """
    For small n (up to 16), use known optimal or near-optimal networks.
    For n=16, use bitonic sort which uses 56 comparators (vs 120 for odd-even transposition).
    """
    if n == 16:
        return _make_bitonic_sort(16)
    elif n == 8:
        return _make_bitonic_sort(8)
    elif n == 4:
        return _make_bitonic_sort(4)
    elif n == 2:
        return _make_bitonic_sort(2)
    elif n == 1:
        return []
    else:
        # Fallback for other sizes
        return _make_odd_even_transposition(n)


def _extend_network(net, n):
    """
    Extend a network for a power-of-2 subset to handle n wires.
    Strategy: Use the optimal network for the largest power of 2 <= n,
    then add extra comparators to handle the remaining wires.
    
    For n=18: use 16-wire bitonic sort (56 comparators) + handle wires 16,17
    """
    # Find largest power of 2 <= n
    p = 1
    while p * 2 <= n:
        p *= 2
    
    base_net = _construct_small_n(p)
    
    if p == n:
        return base_net
    
    # We need to handle wires p through n-1
    # Strategy: 
    # 1. First sort the main p wires
    # 2. Then merge the extra wires into the sorted sequence
    
    # Alternative: Use a different approach
    # For n=18, we can:
    # 1. Sort wires 0-15 with bitonic sort (56 comparators)
    # 2. Insert wires 16,17 into the sorted array
    
    # Actually, let's use a better approach:
    # Use odd-even merge or a custom merge
    
    # For now, let's try: bitonic sort on 16, then merge in the extra 2
    # Merging 2 elements into a sorted array of 16 requires at most ~2*log2(16) = 8 comparators
    # But we need to be careful about correctness
    
    # Better approach: Use a known technique
    # Split into groups and use a merge strategy
    
    # Let's try: sort 0-15 with bitonic, then use a merge network for [0..15] + [16,17]
    # This is complex. Instead, let's use a different strategy.
    
    # Strategy: Use bitonic sort concept extended
    # For n=18, we can think of it as merging sorted subsequences
    
    # Actually, the simplest correct approach:
    # 1. Use bitonic sort for the first 16 wires
    # 2. For wires 16 and 17, do a small sort/merge
    
    # Let's try a different approach: use the fact that we can merge sorted sequences
    
    # For n=18:
    # - Sort wires 0-7 (bitonic, 16 comparators)
    # - Sort wires 8-15 (bitonic, 16 comparators)  
    # - Sort wires 16-17 (1 comparator)
    # - Merge [0-7] and [8-15] (bitonic merge, 16 comparators)
    # - Merge [0-15] and [16-17] (need custom merge)
    
    # This is getting complex. Let's use a simpler but effective approach:
    # Use odd-even transposition for the whole thing but with optimization
    
    # Actually, let's just use bitonic sort for 16 wires and then add
    # comparators to merge in the remaining wires properly
    
    # For merging a sorted array of size m with k extra unsorted elements:
    # We can sort the k extra elements (small), then merge
    
    # Let me try a cleaner approach: use a known good network
    
    # For n=18, let's try:
    # 1. Create two bitonic sorts on halves: [0..8] and [9..17]
    #    But 9 is not a power of 2...
    
    # Let's just use the straightforward approach:
    # Bitonic sort on 16 wires + handle remaining 2 wires with insertion sort
    
    result = list(base_net)
    
    # Sort the remaining wires with each other
    for i in range(p, n):
        for j in range(i + 1, n):
            result.append((i, j))
    
    # Now merge the extra wires into the main sorted array
    # For each extra wire, insert it into the sorted portion
    # This is O(n^2) in worst case but correct
    
    for extra in range(p, n):
        # Insert wire 'extra' into the sorted array [0..p-1]
        # Use a binary search-like insertion
        for i in range(p - 1, -1, -1):
            result.append((i, extra))
            # After comparing, the smaller goes to i, larger to extra
            # We need to shift if extra got a smaller value
    
    # This approach is not quite right for insertion. Let me use a different method.
    
    # Actually, let's use a completely different strategy:
    # Use a known sorting network for n=18 by combining smaller networks
    
    # For now, return the basic approach and verify correctness
    return result


def _try_bitonic_with_extension(n):
    """
    Try to create a bitonic sort network for n wires by extending
    the largest power-of-2 network.
    """
    if n <= 16 and (n & (n - 1)) == 0:
        return _make_bitonic_sort(n)
    
    # For n=18, use a custom approach
    if n == 18:
        # Use bitonic sort on 16 wires
        base = _make_bitonic_sort(16)
        
        # Now we need to incorporate wires 16 and 17
        # Strategy: 
        # 1. Compare wires 16 and 17 to order them
        # 2. Merge the ordered pair into the sorted 16-wire array
        
        net = list(base)
        net.append((16, 17))  # Sort the two extra wires
        
        # Now merge [0..15] (sorted) with [16,17] (sorted)
        # This is a merge of size 16 and size 2
        # We can use a batcher merge or odd-even merge
        
        # Simple merge: for each element in [16,17], insert into [0..15]
        # But we need to do this in a network (parallel)
        
        # Let's use a different approach: 
        # First, merge [0..7] with [8..15] using bitonic merge (already done in base)
        # Then merge [0..15] with [16..17]
        
        # For merging sorted [0..15] with sorted [16,17]:
        # We can think of this as: 
        # - Compare wire 15 with wire 16
        # - Compare wire 14 with wire 16 (shift)
        # - ...
        
        # Actually, let's use a recursive merge approach
        # Merge two sorted arrays of sizes m and k
        
        def merge_sorted(m, k, offset_m=0, offset_k=0):
            """
            Generate comparators to merge two sorted arrays.
            Array 1: wires [offset_m .. offset_m+m-1]
            Array 2: wires [offset_k .. offset_k+k-1]
            Result: merged sorted array in [offset_m .. offset_m+m+k-1]
            """
            if k == 0:
                return []
            if m == 0:
                # Just reorder, but they're already sorted relative to themselves
                return []
            
            if m == 1 and k == 1:
                return [(offset_m, offset_k)]
            
            # Use a divide-and-conquer merge
            # Find the median of the smaller array
            if k < m:
                mid_k = k // 2
                # Compare median of k-array with median of m-array
                # This is complex for a network...
                
                # Simpler: use odd-even merge
                # Or just do a straight-line merge with comparators
                pass
            
            # For small k (like 2), just insert each element
            result = []
            for j in range(k):
                wire_k = offset_k + j
                # Insert wire_k into [offset_m .. offset_m+m-1]
                # Do a shift-insertion
                for i in range(m - 1, -1, -1):
                    wire_m = offset_m + i
                    result.append((wire_m, wire_k))
                    # After this, wire_m has min, wire_k has max
                    # The max might need to be shifted right
                # The final value in wire_k is the largest, which is correct
                # for position m+j
                m += 1
            
            return result
        
        # Merge [0..15] with [16..17]
        merge_net = merge_sorted(16, 2, 0, 16)
        net.extend(merge_net)
        
        return net
    
    # Fallback
    return _make_odd_even_transposition(n)


def _verify_and_fix(net, n):
    """
    Verify the network is correct. If not, fall back to odd-even transposition.
    """
    if len(net) == 0:
        return net
    
    # Quick check: try a few random inputs
    random.seed(42)
    for _ in range(100):
        wires = [random.randint(0, 1) for _ in range(n)]
        for a, b in net:
            if wires[a] > wires[b]:
                wires[a], wires[b] = wires[b], wires[a]
        if any(wires[i] > wires[i + 1] for i in range(n - 1)):
            return _make_odd_even_transposition(n)
    
    # Full verification
    if _is_sorted_network(net):
        return net
    
    return _make_odd_even_transposition(n)


def construct():
    """
    Construct a sorting network for N wires.
    Returns a list of (i, j) pairs where i < j, representing comparators.
    """
    start_time = time.time()
    
    # Try the bitonic-with-extension approach
    candidate = _try_bitonic_with_extension(N)
    
    # Verify it's correct
    if time.time() - start_time < TIME_LIMIT - 5:
        verified = _verify_and_fix(candidate, N)
    else:
        verified = _make_odd_even_transposition(N)
    
    # Safety check: ensure we're within time limit
    if time.time() - start_time > TIME_LIMIT - 2:
        return _make_odd_even_transposition(N)
    
    return verified
