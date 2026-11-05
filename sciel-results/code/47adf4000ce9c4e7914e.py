import time
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N
TIME_LIMIT = 55.0  # Safety margin

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Batcher's Odd-Even Merge Sort
    # Returns a list of (i, j) comparators
    
    def oe_merge(n, offset=0, stride=1):
        """
        Performs Odd-Even Merge on n wires starting at 'offset' with 'stride'.
        Assumes the n wires contain two sorted sequences of length n/2 interleaved?
        No, OE_Merge(n) merges two sorted lists of length n/2.
        The inputs are assumed to be in wires:
        offset, offset+stride, ..., offset+(n-1)*stride
        The first half (0 to n/2-1) is sorted.
        The second half (n/2 to n-1) is sorted.
        """
        if n <= 1:
            return []
        
        comps = []
        m = n // 2
        
        # Step 1: OE_Sort on odd wires (indices 0, 2, 4... in the local n-frame)
        # These correspond to physical wires: offset, offset+2*stride, ...
        # There are m such wires.
        comps.extend(oe_sort(m, offset, 2 * stride))
        
        # Step 2: OE_Sort on even wires (indices 1, 3, 5... in the local n-frame)
        # Physical wires: offset+stride, offset+3*stride, ...
        comps.extend(oe_sort(m, offset + stride, 2 * stride))
        
        # Step 3: OE_Merge on the whole set?
        # No, Step 3 is a sequence of comparators:
        # For i in range(m):
        #    Compare (i, m+i)
        # Physical wires:
        # i -> offset + i * stride
        # m+i -> offset + (m+i) * stride
        
        for i in range(m):
            wire1 = offset + i * stride
            wire2 = offset + (m + i) * stride
            if wire1 < wire2:
                comps.append((wire1, wire2))
            else:
                comps.append((wire2, wire1))
                
        return comps

    def oe_sort(n, offset=0, stride=1):
        """
        Performs Odd-Even Sort on n wires starting at 'offset' with 'stride'.
        """
        if n <= 1:
            return []
        
        comps = []
        m = n // 2
        
        if n % 2 == 0:
            # 1. OE_Sort(m) on first half
            comps.extend(oe_sort(m, offset, stride))
            # 2. OE_Sort(m) on second half
            comps.extend(oe_sort(m, offset + m * stride, stride))
            # 3. OE_Merge(m) on the whole array (n wires)
            # Wait, OE_Merge takes two sorted lists of size m.
            # The result of steps 1 and 2 are two sorted lists of size m.
            # So we call OE_Merge(n, offset, stride)
            comps.extend(oe_merge(n, offset, stride))
        else:
            # n is odd
            # 1. OE_Sort(m) on first half (size m)
            comps.extend(oe_sort(m, offset, stride))
            # 2. OE_Sort(m+1) on second half (size m+1)
            comps.extend(oe_sort(m + 1, offset + m * stride, stride))
            # 3. OE_Merge(m) on the first n-1 wires?
            # The last wire (offset + (n-1)*stride) is the max of the second half.
            # We merge the first m wires of the first half with the first m wires of the second half.
            # The comparators are between i and m+i for i in 0..m-1.
            # This is effectively OE_Merge(m) on the wires:
            # offset + i*stride and offset + (m+i)*stride
            # But OE_Merge expects a contiguous block of 2*m wires.
            # Here the wires are:
            # First half: offset ... offset + (m-1)*stride
            # Second half part: offset + m*stride ... offset + (2m-1)*stride
            # These are contiguous in the logical sense if stride=1.
            # With stride, they are not contiguous in physical memory, but the logic holds.
            # We can just generate the comparators directly as in oe_merge, but only for m pairs.
            
            # Actually, oe_merge(n) generates comparators for i in 0..m-1 between i and m+i.
            # If we call oe_merge(n, offset, stride), it will also do the recursive sorts inside.
            # We already did the sorts. We just need the final merge comparators.
            
            # Let's create a helper for just the comparators of the merge step.
            # Or just call oe_merge and ignore the recursive parts? No, that's inefficient.
            
            # Let's inline the merge comparators.
            for i in range(m):
                wire1 = offset + i * stride
                wire2 = offset + (m + i) * stride
                if wire1 < wire2:
                    comps.append((wire1, wire2))
                else:
                    comps.append((wire2, wire1))
            
        return comps

    # Generate the network for N wires
    network = oe_sort(N, 0, 1)
    
    # Optional: Greedy removal to improve size
    # Verify correctness on a sample first to ensure we don't break it
    # Then verify on all inputs
    
    def verify(network, sample_size=None):
        if sample_size is None:
            inputs = np.arange(TOTAL_INPUTS, dtype=np.int64)
        else:
            inputs = np.random.randint(0, TOTAL_INPUTS, sample_size, dtype=np.int64)
            
        states = inputs.copy()
        for i, j in network:
            mask_i = 1 << i
            mask_j = 1 << j
            inv_mask = (states & mask_i) & ~(states & mask_j)
            states[inv_mask] = states[inv_mask] ^ mask_i ^ mask_j
            
        # Check if sorted
        # Sorted means bits are 0...01...1
        # For each state, count set bits. The set bits should be at the highest indices.
        # Mask for sorted state with k ones: ((1 << k) - 1) << (N - k)
        # We can check: state == ((1 << popcount(state)) - 1) << (N - popcount(state))
        
        # Vectorized check
        # popcount for int64 array
        # np.bitwise_count is not in older numpy, use bin(x).count('1')? Too slow.
        # Use a lookup table for popcount for 18 bits? 2^18 = 262k.
        # Or just check the pattern.
        
        # A state is sorted if for all k < l, if bit k is 1 then bit l is 1.
        # Equivalent: state & (state - 1) == 0? No, that's power of 2.
        # Equivalent: state == ((1 << k) - 1) << (N - k) for some k.
        
        # Let's precompute sorted masks
        sorted_masks = set()
        for k in range(N + 1):
            sorted_masks.add(((1 << k) - 1) << (N - k))
            
        # Check each state
        # This is slow in Python loop.
        # Use a boolean array lookup.
        is_sorted_arr = np.zeros(TOTAL_INPUTS, dtype=bool)
        for m in sorted_masks:
            is_sorted_arr[m] = True
            
        return np.all(is_sorted_arr[states])

    # If we have time, try to remove comparators
    # 110 comparators.
    # Removing 1 comp and verifying on 1000 samples is fast.
    # Verifying on all inputs is ~0.1-0.5s.
    # We have 55s.
    # 110 removals * 0.5s = 55s. Tight.
    # Let's try a few removals.
    
    # First, verify the base network is correct (it should be)
    # print("Verifying base network...")
    # if not verify(network):
    #     return network # Fallback to unoptimized if bug in verification
    
    # Greedy Removal
    # We try to remove each comparator. If the network remains correct, we keep it removed.
    # To speed up, we first check on a small sample.
    
    current_network = network.copy()
    removed_indices = []
    
    # Pre-compute is_sorted_arr once
    sorted_masks = set()
    for k in range(N + 1):
        sorted_masks.add(((1 << k) - 1) << (N - k))
    is_sorted_arr = np.zeros(TOTAL_INPUTS, dtype=bool)
    for m in sorted_masks:
        is_sorted_arr[m] = True
        
    def verify_fast(net):
        states = np.arange(TOTAL_INPUTS, dtype=np.int64)
        for i, j in net:
            mask_i = 1 << i
            mask_j = 1 << j
            inv_mask = (states & mask_i) & ~(states & mask_j)
            states[inv_mask] = states[inv_mask] ^ mask_i ^ mask_j
        return np.all(is_sorted_arr[states])
    
    # Try removing comparators
    # We iterate backwards to avoid index shifting issues, or use a list of indices to remove.
    # Simpler: iterate through indices, if removal is valid, mark for removal.
    # But removal changes the network for subsequent checks.
    # So we must re-verify after each successful removal.
    
    i = 0
    while i < len(current_network):
        if time.time() - start_time > TIME_LIMIT - 5: # Keep 5s buffer
            break
            
        # Try removing current_network[i]
        test_net = current_network[:i] + current_network[i+1:]
        
        # Quick check on sample
        sample_states = np.arange(1000, dtype=np.int64) # Just use first 1000 for speed?
        # Better: random sample
        sample_states = np.random.randint(0, TOTAL_INPUTS, 1000, dtype=np.int64)
        
        s = sample_states.copy()
        for k, l in test_net:
            mask_k = 1 << k
            mask_l = 1 << l
            inv_mask = (s & mask_k) & ~(s & mask_l)
            s[inv_mask] = s[inv_mask] ^ mask_k ^ mask_l
            
        if not np.all(is_sorted_arr[s]):
            i += 1
            continue
            
        # Full verification
        if verify_fast(test_net):
            current_network = test_net
            # Do not increment i, check the new current_network[i] next
        else:
            i += 1
            
    return current_network
# EVOLVE-BLOCK-END
