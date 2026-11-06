import numpy as np
import time

N = 16

def verify_network(comps, N=16):
    """
    Verifies if the given comparator network sorts all 2^N binary inputs.
    Uses the Zero-One Principle.
    """
    if not comps:
        # Empty network only sorts if N <= 1
        return N <= 1
        
    # Create input matrix: 16 rows, 2^16 columns
    # wires[i, k] is the value of wire i for input k
    # Input k is a 16-bit integer.
    # wire 0 is MSB (bit 15), wire 15 is LSB (bit 0).
    
    k = np.arange(1 << N, dtype=np.uint16)
    
    # Initialize wires
    # wires[i] = (k >> (15 - i)) & 1
    wires = np.empty((N, 1 << N), dtype=np.uint8)
    for i in range(N):
        wires[i] = (k >> (N - 1 - i)) & 1
        
    # Apply comparators
    for i, j in comps:
        # min to i, max to j
        # For 0/1 values: min = AND, max = OR
        min_val = wires[i] & wires[j]
        max_val = wires[i] | wires[j]
        wires[i] = min_val
        wires[j] = max_val
        
    # Check if sorted
    # For each column, values must be non-decreasing
    # i.e., 0s then 1s.
    # np.diff(wires, axis=0) should be >= 0
    diffs = np.diff(wires, axis=0)
    return np.all(diffs >= 0)

def get_bitonic_16():
    """
    Returns the standard 80-comparator Bitonic Sorting Network for 16 inputs.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Start with Bitonic 80
    comps = get_bitonic_16()
    
    # Verify initial network (sanity check)
    if not verify_network(comps, N):
        # Fallback to seed if Bitonic is somehow broken
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Greedy Removal
    # Iterate from the end of the list to the beginning
    # If removing a comparator keeps the network valid, remove it.
    # We do this in a loop until no more can be removed.
    # Since removing a comparator cannot invalidate a previously valid network,
    # a single pass from end to start is sufficient for a local minimum 
    # with respect to single-comparator removals in that order.
    # However, removing one might make another removable that wasn't before?
    # No, if A is redundant in N, and B is redundant in N\{A}, 
    # it doesn't mean B was redundant in N.
    # So we should repeat the process until no change occurs.
    
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() > deadline:
            break
            
        # Iterate backwards to handle index shifts safely
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Try removing comps[idx]
            # Create new list without idx
            # Optimization: slicing creates a new list. 
            # For 80 items, this is cheap.
            candidate = comps[:idx] + comps[idx+1:]
            
            if verify_network(candidate, N):
                comps = candidate
                improved = True
                # Continue loop with the new shorter list
                # Note: The for loop range is fixed at the start of the while iteration.
                # If we remove an item, the indices of the remaining items 
                # in the current for-loop iteration might be off?
                # No, we are iterating backwards.
                # If we remove idx, the items at 0..idx-1 are unchanged.
                # The next index to check is idx-1.
                # In the new list, the item at idx-1 is still at idx-1.
                # So the for loop continues correctly.
                pass
            else:
                # Cannot remove
                pass
                
    # Final verification
    if not verify_network(comps, N):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return comps
