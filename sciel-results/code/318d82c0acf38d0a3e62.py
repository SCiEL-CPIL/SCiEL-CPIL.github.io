import time
import random
import numpy as np

N = 16
DEADLINE = time.time() + 85

def _batcher_odd_merge_network(n):
    """
    Generates Batcher's odd-even merge sort network for n wires (power of 2).
    Returns a list of (i, j) comparators.
    """
    def generate(start, length):
        if length <= 1:
            return []
        half = length // 2
        comps = []
        
        # Recursively sort both halves
        comps.extend(generate(start, half))
        comps.extend(generate(start + half, half))
        
        # Odd-even merge: interleave and compare
        # For odd-even merge of two sorted lists of size half:
        # Compare elements at positions (2k) and (2k+1) in the merged sequence
        # The merge pattern for size m = half:
        for k in range(half):
            # Compare (start + 2*k) with (start + 2*k + 1) if both in range
            # Actually, odd-even merge works by recursively merging pairs
            pass
        
        # Simpler approach: use the standard odd-even merge algorithm
        # For two sorted sequences of size m, odd-even merge:
        # Step 1: Compare (0,1), (2,3), (4,5), ... (odd-even pairs)
        # Step 2: Compare (1,2), (3,4), (5,6), ... (even-odd pairs)  
        # But this is for the merge of two already-sorted halves
        
        # Actually, let me implement it properly using the recursive structure
        return _odd_even_merge(start, half, length)
    
    def _odd_even_merge(start, m, total_length):
        """Odd-even merge of two sorted sequences of size m starting at start."""
        if m <= 1:
            return []
        half = m // 2
        comps = []
        
        # Recursively odd-even merge the first half of each sequence
        # and the second half of each sequence
        # First sequence: start to start+m-1
        # Second sequence: start+m to start+2*m-1
        
        # Odd-even merge step:
        # 1. Compare (start+i, start+m+i) for i in 0..m-1 (cross-compare)
        # 2. Then recursively merge the resulting two sorted sequences
        
        # Standard odd-even merge:
        # Compare elements: (0, m), (1, m+1), ..., (m-1, 2m-1)
        for i in range(m):
            a = start + i
            b = start + m + i
            comps.append((a, b))
        
        # Now we have two sorted sequences of size m
        # Recursively merge them using odd-even merge
        # The two sequences are now at start..start+m-1 and start+m..start+2m-1
        # But after the cross-compare, they form a bitonic sequence
        # We need to sort this bitonic sequence
        
        # Actually, odd-even merge is:
        # 1. Split into two halves
        # 2. Recursively odd-even merge each half
        # 3. Interleave
        
        # Let me use a cleaner recursive definition
        return _odd_even_merge_recursive(start, m)
    
    def _odd_even_merge_recursive(start, m):
        """Recursively odd-even merge two sorted sequences of size m."""
        if m <= 1:
            return []
        half = m // 2
        comps = []
        
        # First, compare (start+i, start+m+i) for i in 0..m-1
        for i in range(m):
            a = start + i
            b = start + m + i
            comps.append((a, b))
        
        # Now the result is a bitonic sequence of size 2*m
        # Sort it using bitonic sort's merge step
        # This is equivalent to: sort first half, sort second half, then merge
        # But for odd-even merge, we recursively merge the two halves
        
        # After cross-compare, we have two sorted sequences:
        # Sequence 1: start to start+m-1 (sorted)
        # Sequence 2: start+m to start+2*m-1 (sorted)
        # We need to merge these two sorted sequences
        
        # Odd-even merge of two sorted sequences of size m:
        # If m is even, we can split into two problems of size m/2
        # Compare (start+i, start+m+i) for i=0..m-1, then recursively merge
        
        # The recursive structure:
        # 1. Split each sequence into two halves of size half
        # 2. Recursively odd-even merge the corresponding halves
        # 3. The result is sorted
        
        # Actually, the standard odd-even merge algorithm:
        # To merge two sorted sequences of size m:
        # 1. Compare (i, m+i) for i = 0, 1, ..., m-1
        # 2. This produces two sorted sequences of size m
        # 3. Recursively merge these two sequences
        
        # But step 1 already gives us the cross-comparisons
        # After step 1, we have two sorted sequences
        # We need to merge them, which is the same problem but...
        
        # I think I'm overcomplicating this. Let me use a known correct implementation.
        return comps
    
    # Let me restart with a cleaner implementation
    def _merge_sorted(a_start, b_start, m):
        """Merge two sorted sequences of size m starting at a_start and b_start."""
        if m <= 1:
            return []
        half = m // 2
        comps = []
        
        # Cross-compare: (a_start+i, b_start+i) for i in 0..m-1
        for i in range(m):
            a = a_start + i
            b = b_start + i
            comps.append((a, b))
        
        # Now we have two sorted sequences of size m
        # Recursively merge them
        # The two sequences are at a_start..a_start+m-1 and b_start..b_start+m-1
        # But after cross-compare, they form a bitonic sequence
        # We need to sort this, which is:
        # 1. Sort first half (a_start to a_start+m-1) - already sorted
        # 2. Sort second half (b_start to b_start+m-1) - already sorted  
        # 3. Merge them - this is the same problem!
        
        # Actually, after cross-compare of two sorted sequences, the result is
        # a bitonic sequence. To sort a bitonic sequence of size 2m:
        # 1. Split into two halves of size m
        # 2. Sort first half ascending, second half descending
        # 3. Compare (i, m+i) for i=0..m-1
        # 4. Recursively sort the two halves
        
        # But that's bitonic sort, not odd-even merge.
        
        # Let me just use bitonic sort as a reliable starting point
        return comps
    
    return _generate_bitonic(0, n)

def _generate_bitonic(start, length, direction=1):
    """Generate bitonic sort network."""
    if length <= 1:
        return []
    half = length // 2
    comps = []
    
    # Sort first half ascending, second half descending
    comps.extend(_generate_bitonic(start, half, 1))
    comps.extend(_generate_bitonic(start + half, half, -1))
    
    # Bitonic merge
    for i in range(half):
        a = start + i
        b = start + half + i
        if direction == 1:
            comps.append((a, b))
        else:
            comps.append((b, a))
    
    comps.extend(_generate_bitonic(start, half, direction))
    comps.extend(_generate_bitonic(start + half, half, direction))
    
    return comps

def _verify_full(network):
    """
    Full verification using numpy over all 2^N states.
    Returns True if the network sorts all inputs.
    """
    states = np.arange(1 << N, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        # If bi > bj, we need to swap to put min at i and max at j
        swap = bi > bj
        # Flip bits i and j where swap is True
        mask = (1 << i) | (1 << j)
        states = np.where(swap, states ^ mask, states)
    
    # Check if all states are sorted (0s followed by 1s)
    # A number is sorted if (state + 1) & state == 0
    return bool(np.all(((states + 1) & states) == 0))

def _verify_sample(network, n_samples=10000, rng=None):
    """
    Sampled verification for speed during search.
    """
    if rng is None:
        rng = random.Random(42)
    for _ in range(n_samples):
        state = rng.randrange(1 << N)
        s = state
        for i, j in network:
            if ((s >> i) & 1) > ((s >> j) & 1):
                s ^= (1 << i) | (1 << j)
        if (s + 1) & s != 0:
            return False
    return True

def construct():
    start_time = time.time()
    
    # 1. Generate Bitonic Sort Network (56 comparators for N=16)
    net = _generate_bitonic(0, N, 1)
    best = list(net)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback: Odd-Even Transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    rng = random.Random(12345)
    
    # 2. Greedy Removal with full verification (faster than sampled for N=16)
    # Full verification over 65536 states is fast with numpy
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > DEADLINE - 5:
                break
            
            candidate = best[:idx] + best[idx+1:]
            
            # Use full verification - it's fast enough with numpy
            if _verify_full(candidate):
                best = candidate
                improved = True
    
    # Final full verification
    if _verify_full(best):
        return best
    else:
        # If the greedy search corrupted the network, fall back to the initial valid one
        initial_net = _generate_bitonic(0, N, 1)
        if _verify_full(initial_net):
            return initial_net
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
