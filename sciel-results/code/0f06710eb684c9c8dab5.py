import time
import random

def construct():
    n = 512
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    # Precompute 4-APs and conflict masks
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We use 0-indexed indices for the bitmask
    # Element i corresponds to index i
    
    # other_masks[i] will be a list of bitmasks representing the other 3 elements
    # for each 4-AP that contains i.
    other_masks = [[] for _ in range(n)]
    
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(1, n - 3 * d + 1):
            # 1-indexed AP: a, a+d, a+2d, a+3d
            # 0-indexed indices:
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            # For each element in the AP, store the mask of the other 3
            # For i0: others are i1, i2, i3
            m0 = (1 << i1) | (1 << i2) | (1 << i3)
            other_masks[i0].append(m0)
            
            # For i1: others are i0, i2, i3
            m1 = (1 << i0) | (1 << i2) | (1 << i3)
            other_masks[i1].append(m1)
            
            # For i2: others are i0, i1, i3
            m2 = (1 << i0) | (1 << i1) | (1 << i3)
            other_masks[i2].append(m2)
            
            # For i3: others are i0, i1, i2
            m3 = (1 << i0) | (1 << i1) | (1 << i2)
            other_masks[i3].append(m3)

    def can_add(i, mask):
        """
        Check if adding index i to the current mask creates a 4-AP.
        Returns True if valid, False if invalid.
        """
        for om in other_masks[i]:
            if (mask & om) == om:
                return False
        return True

    best_mask = 0
    best_size = 0
    
    # Run multiple randomized greedy restarts
    # Each restart tries to build a maximal 4-AP-free set by picking elements
    # in a random order.
    
    while time.time() < deadline:
        mask = 0
        # Create a list of indices and shuffle
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if can_add(i, mask):
                mask |= (1 << i)
        
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
        # Optional: Local search to see if we can improve this specific mask
        # Try to add any remaining valid elements (greedy already did this in its order, 
        # but maybe a different order of checking remaining elements helps? 
        # Actually, greedy added ALL valid elements in the shuffled order. 
        # So the set is maximal with respect to the order. 
        # No simple "add" step will work. 
        # We could try swaps, but that's expensive. 
        # Let's stick to restarts for now.)
        
    # Convert best_mask to sorted list of 1-based integers
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)
