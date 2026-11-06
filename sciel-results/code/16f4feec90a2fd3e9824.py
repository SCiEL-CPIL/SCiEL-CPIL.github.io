"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _batcher_sort_network(n):
    """Generate a Batcher odd-even mergesort network for n wires."""
    if n == 1:
        return []
    
    if n % 2 == 1:
        # Recurse on n-1, then add comparators for the new wire
        sub = _batcher_sort_network(n - 1)
        # The new wire n-1 must be merged into the sorted n-1 wires
        # Batcher's approach for odd: use the even part to sort, then merge
        # Simpler: just use the standard recursive definition
        # For odd n: network = network(n-1) + comparators to insert wire n-1
        # Actually, let me use the standard construction
        # Batcher OES: if n is odd, sort n-1 wires, then merge the single wire
        # But the standard construction is cleaner for powers of 2
        # Let's use a different approach for odd: 
        # Split into even/odd halves, sort each, then merge
        # For simplicity and correctness, let me implement the standard recursive Batcher
        
        # Standard Batcher:
        # 1. If n is power of 2, use standard OES
        # 2. If not, recurse and then merge
        
        # Actually, let me just implement the full recursive Batcher properly
        # For now, let's handle it by padding to next power of 2 and then removing
        # But that's not optimal. Let me implement the proper recursion.
        
        # Proper Batcher recursion for arbitrary n:
        # If n is even: split into two halves of n/2, sort each, then merge
        # If n is odd: split into floor(n/2) and ceil(n/2), sort each, then merge
        
        half1 = n // 2
        half2 = n - half1
        
        # Sort the two halves
        net1 = _batcher_sort_network(half1)
        net2 = _batcher_sort_network(half2)
        # net2 needs to be offset by half1
        net2_offset = [(i + half1, j + half1) for (i, j) in net2]
        
        # Merge the two sorted halves
        merge_net = _batcher_merge_network(half1, half2)
        merge_net_offset = [(i, j) for (i, j) in merge_net]
        # merge_net should already have the right indices if implemented correctly
        # Let me implement merge separately
        
        return net1 + net2_offset + merge_net_offset
    
    # n is even
    half = n // 2
    net1 = _batcher_sort_network(half)
    net2 = _batcher_sort_network(half)
    net2_offset = [(i + half, j + half) for (i, j) in net2]
    merge_net = _batcher_merge_network(half, half)
    return net1 + net2_offset + merge_net


def _batcher_merge_network(m, k):
    """Generate Batcher merge network to merge two sorted lists of sizes m and k.
    m <= k. The first m wires contain the first sorted list, the next k wires the second."""
    if m == 0:
        return []
    if k == 0:
        return []
    
    # Batcher merge: 
    # Step 1: Compare-exchange (i, m+i) for i in range(min(m, k)) ... actually
    # The standard Batcher merge for two sorted sequences of length m and k (m <= k):
    # 1. For i in 0..min(m,k)-1: compare (i, m+i)
    # 2. Recursively merge the two resulting sequences
    
    # Actually, the standard construction is:
    # To merge two sorted lists of size m and k where m <= k:
    # 1. Compare (i, m+i) for i = 0 to m-1
    # 2. This gives two sequences: 
    #    A = first m elements (from positions 0..m-1 after compare)
    #    B = remaining k elements (from positions m..m+k-1 after compare)
    # 3. Recursively merge A and B... but A and B are not contiguous in the standard way
    
    # Let me use the well-known recursive formula:
    # merge(m, k) for m <= k:
    #   if m == 0 or k == 0: return []
    #   Step 1: comparators (i, m+i) for i in range(m)  [but only where m+i < m+k]
    #   Step 2: The result can be viewed as merging:
    #     - Left part: merge of (first floor(m/2) of A) and (first ceil(k/2) of B) 
    #     - This is getting complicated. Let me use a known correct implementation.
    
    # Known correct Batcher merge:
    # To merge sorted lists of sizes a and b (a <= b) on wires 0..a+b-1:
    # 1. For i in 0..a-1: compare (i, a+i)
    # 2. Recursively: 
    #    - Merge wires 0..ceil((a+b)/2)-1 (which now contain the smaller elements)
    #    - Merge wires floor((a+b)/2)..a+b-1 (which now contain the larger elements)
    
    total = m + k
    if total <= 1:
        return []
    
    result = []
    
    # Step 1: cross comparators
    for i in range(m):
        j = m + i
        if j < total:
            result.append((i, j))
    
    # Step 2: recursive merge
    # After step 1, the smaller elements tend to be in the first half
    half_total = total // 2
    
    # The first half (0..half_total-1) needs to be merged
    # It contains min(half_total, m) elements from the first list and 
    # max(0, half_total - m) elements from the second list... 
    # Actually, after the cross comparators, wire i (i < m) has min(A[i], B[i])
    # and wire m+i has max(A[i], B[i]).
    # The elements in positions 0..half_total-1 form a merge of two sorted sequences.
    # The first sequence has size min(m, half_total) 
    # The second sequence has size max(0, half_total - m)
    # Wait, this isn't quite right. Let me think more carefully.
    
    # After step 1:
    # For i in 0..m-1: wire i = min(A[i], B[i]), wire m+i = max(A[i], B[i])
    # For i in m..m+k-1 where i >= m+m (i.e., i >= 2m): wire i = B[i-m] (unchanged)
    
    # The set of wires 0..half_total-1 contains:
    # - For i in 0..min(m, half_total)-1: min(A[i], B[i])
    # - For i in m..half_total-1 (if half_total > m): max(A[i-m], B[i-m]) 
    #   But wait, these max values are larger, so they might not all be in the first half
    
    # This is getting complex. Let me use a simpler, known-correct approach.
    
    # Alternative: Use the fact that after step 1, we can recursively merge:
    # merge(min(m, half_total), max(0, half_total - m)) on wires 0..half_total-1
    # merge(max(0, total - half_total - k + (k - max(0, total - half_total - m))), ...) 
    # This is error-prone. Let me just implement it directly with a known formula.
    
    # Actually, the cleanest way: after step 1, the problem decomposes into:
    # - Merging the "small" parts: size min(m, total//2) and max(0, total//2 - m)  
    #   on wires 0..total//2-1
    # - Merging the "large" parts: size max(0, total - total//2 - k) and 
    #   min(k, total - total//2) on wires total//2..total-1
    
    # Let me just hardcode a correct implementation by using the standard 
    # recursive Batcher merge that is well-documented.
    
    # For the merge step, after the initial comparators (i, m+i) for i in range(m):
    # We recursively merge:
    #   Left: sizes min(m, half_total) and max(0, half_total - m), on wires [0, half_total)
    #   Right: sizes max(0, total - half_total - (k - min(k, total - half_total))) 
    #           and min(k, total - half_total), on wires [half_total, total)
    
    # Let me simplify: 
    # left_size1 = min(m, half_total)
    # left_size2 = max(0, half_total - m)  # this is the part from the second list
    # right_size1 = max(0, (m - half_total))  # no wait
    
    # I'll use a different, cleaner formulation. The standard Batcher merge:
    # def merge(a, b):  # a <= b
    #   if a == 0 or b == 0: return []
    #   net = [(i, a+i) for i in range(a)]
    #   h = (a+b) // 2
    #   net += merge(min(a, h), max(0, h-a))  # on first h wires
    #   net += merge(max(0, a+b-h-b), min(b, a+b-h))  # on last a+b-h wires, offset by h
    #   return net
    # But the second recursive call needs offsetting.
    
    # Let me implement with explicit offsets.
    
    if m == 0 or k == 0:
        return []
    
    net = []
    # Initial cross comparators
    for i in range(m):
        if m + i < total:
            net.append((i, m + i))
    
    half_total = total // 2
    
    # Left sub-merge
    left_a = min(m, half_total)
    left_b = max(0, half_total - m)
    if left_a > 0 and left_b > 0:
        left_sub = _batcher_merge_network(left_a, left_b)
        net.extend(left_sub)
    elif left_a > 0:
        pass  # Already sorted
    elif left_b > 0:
        pass  # Already sorted
    
    # Right sub-merge
    right_start = half_total
    right_total = total - half_total
    # The right part has:
    # From the first list: max(0, m - half_total) elements
    # From the second list: max(0, k - max(0, half_total - m)) elements
    # But we need to be careful about what's in the right half.
    
    # After the initial comparators:
    # Wires 0..m-1: min(A[i], B[i]) for i in 0..m-1
    # Wires m..m+k-1: max(A[i-m], B[i-m]) for i-m in 0..m-1 (i.e., m..2m-1), 
    #                  and B[i-m] for i-m in m..k-1 (i.e., 2m..m+k-1)
    
    # The right half (half_total..total-1) contains:
    # - Some of the max(A[i], B[i]) values
    # - Some of the unchanged B[j] values
    
    # For the recursive merge, the right half contains a merge of:
    # - Size max(0, m - half_total) from the "max" part of the first list
    # - Size max(0, k - max(0, half_total - m)) from the second list
    #   But this isn't quite right either.
    
    # Let me use a simpler approach: just verify with the 0-1 principle.
    # For now, let me use a well-tested implementation.
    
    # I'll use the following known-correct recursive structure:
    # After step 1 comparators, recursively:
    # - Merge the first h wires as two sorted sequences of sizes 
    #   min(m, h) and max(0, h - m)
    # - Merge the last total-h wires as two sorted sequences of sizes
    #   max(0, m - h) and max(0, k - (h - min(m, h)))
    #   Wait, that second term: the number of elements from the second list 
    #   that ended up in the right half.
    
    # Let me just compute:
    # Elements from list A (size m) in left half: min(m, half_total)
    # Elements from list A in right half: m - min(m, half_total) = max(0, m - half_total)
    # Elements from list B (size k) in left half: max(0, half_total - m)
    # Elements from list B in right half: k - max(0, half_total - m) = max(0, k - half_total + m)
    
    right_a = max(0, m - half_total)
    right_b = max(0, k - (half_total - min(m, half_total)))
    
    if right_a > 0 and right_b > 0:
        right_sub = _batcher_merge_network(right_a, right_b)
        right_sub_offset = [(i + half_total, j + half_total) for (i, j) in right_sub]
        net.extend(right_sub_offset)
    
    return net


def _verify_network(network, n):
    """Verify network sorts all 2^n boolean inputs using NumPy."""
    # Generate all 2^n boolean vectors
    num_vectors = 1 << n
    # Create all combinations
    bits = np.array(list(np.ndindex(*([2] * n))), dtype=np.uint8)
    bits = bits.reshape(-1, n)
    
    # Apply network
    for (i, j) in network:
        a = bits[:, i]
        b = bits[:, j]
        bits[:, i] = np.minimum(a, b)
        bits[:, j] = np.maximum(a, b)
    
    # Check if sorted
    for col in range(n - 1):
        if np.any(bits[:, col] > bits[:, col + 1]):
            return False
    return True


def _remove_redundant(network, n, max_time=45.0):
    """Try to remove redundant comparators via greedy elimination."""
    start = time.time()
    current = list(network)
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start
        if elapsed > max_time:
            break
        
        for idx in range(len(current)):
            if time.time() - start > max_time:
                break
            
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            
            # Quick check: is the network still valid?
            # Use full verification
            if _verify_network(candidate, n):
                current = candidate
                improved = True
                break  # Restart from beginning after a removal
    
    return current


def _random_check(network, n, num_trials=10000, seed=42):
    """Random check with specific values (not just 0/1) for additional safety."""
    rng = np.random.RandomState(seed)
    for _ in range(num_trials):
        vec = rng.rand(n)
        vec = list(vec)
        for (i, j) in network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        for k in range(n - 1):
            if vec[k] > vec[k + 1]:
                return False
    return True


# EVOLVE-BLOCK-END


def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # Generate Batcher network for N=18
    network = _batcher_sort_network(N)
    
    # Verify the initial network
    if not _verify_network(network, N):
        # Fallback to odd-even transposition if Batcher fails
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Verify with random checks too
    if not _random_check(network, N):
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Try to optimize by removing redundant comparators
    if time.time() < deadline - 30.0:
        network = _remove_redundant(network, N, max_time=30.0)
    
    # Final verification
    if not _verify_network(network, N):
        # Fall back to a known-correct network
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Ensure within limits
    if len(network) > 4096:
        network = network[:4096]
    
    return network
