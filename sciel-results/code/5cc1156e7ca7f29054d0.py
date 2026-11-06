"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _check_network(network, full_check=True, sample_size=1000):
    """
    Check if the network sorts all inputs.
    Uses 0-1 principle.
    """
    if full_check:
        indices = np.arange(2**N)
        inputs = ((indices[:, None] >> np.arange(N)[None, :]) & 1).astype(int)
        
        for i, j in network:
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
            
        diffs = np.diff(inputs, axis=1)
        if np.any(diffs < 0):
            return False
        return True
    else:
        samples = np.random.randint(0, 2, size=(sample_size, N))
        for i, j in network:
            min_val = np.minimum(samples[:, i], samples[:, j])
            max_val = np.maximum(samples[:, i], samples[:, j])
            samples[:, i] = min_val
            samples[:, j] = max_val
        diffs = np.diff(samples, axis=1)
        if np.any(diffs < 0):
            return False
        return True

def _bitonic_sort(n):
    """
    Generate Batcher's Bitonic Sort network for n wires (n must be power of 2).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    comps = []
    
    # Helper to generate bitonic merge
    def bitonic_merge(n, offset, direction):
        # n is the size of the block being merged
        # direction: 1 for ascending, -1 for descending
        if n == 1:
            return
        half = n // 2
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 1:
                if a > b: a, b = b, a
            else:
                if a < b: a, b = b, a
            comps.append((a, b))
        # Recursively sort the two halves
        bitonic_merge(half, offset, 1) # First half ascending
        bitonic_merge(half, offset + half, -1 if direction == 1 else 1) # Second half opposite?
        # Wait, standard bitonic merge:
        # If we are merging two bitonic sequences (one asc, one desc) into one sorted sequence:
        # The comparators are (i, i+n/2).
        # Then we recursively sort the first n/2 and second n/2.
        # The direction of the recursive sort depends on the desired output direction.
        
        # Let's use a cleaner recursive definition for Bitonic Sort.
        
    # Standard Bitonic Sort Algorithm
    def generate_bitonic_sort(n, offset, direction):
        if n <= 1:
            return
        if n == 2:
            a = offset
            b = offset + 1
            if direction == 1:
                if a > b: a, b = b, a
            else:
                if a < b: a, b = b, a
            comps.append((a, b))
            return
            
        half = n // 2
        # Sort first half in ascending order
        generate_bitonic_sort(half, offset, 1)
        # Sort second half in descending order
        generate_bitonic_sort(half, offset + half, -1)
        
        # Merge the two halves
        # The merge step compares (i, i+half) for i in 0..half-1
        # But we need to do this in a specific depth structure.
        # Actually, the standard recursive call for merge is:
        # for i in range(half):
        #    comps.append((offset+i, offset+i+half))
        # Then recursively sort the resulting halves?
        # No, the merge step itself is a network of depth log(n).
        
        # Let's use the explicit loop for merge which is equivalent to recursive calls of depth.
        # Bitonic Merge of size n:
        # Depth 1: compare (0, n/2), (1, n/2+1), ...
        # Then recursively merge the two halves of size n/2.
        
        def merge(n, offset, direction):
            if n == 1:
                return
            half = n // 2
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 1:
                    if a > b: a, b = b, a
                else:
                    if a < b: a, b = b, a
                comps.append((a, b))
            merge(half, offset, 1)
            merge(half, offset + half, 1)
            
        # To sort n elements:
        # 1. Split into two halves.
        # 2. Sort first half ascending.
        # 3. Sort second half descending.
        # 4. Merge them.
        
        generate_bitonic_sort(half, offset, 1)
        generate_bitonic_sort(half, offset + half, -1)
        merge(n, offset, direction)

    generate_bitonic_sort(n, 0, 1)
    return comps

def construct():
    """
    Construct a sorting network for 17 wires.
    Uses Bitonic Sort for the first 16 wires and a 1-in-16 merge for the 17th.
    """
    # Sort first 16 wires
    network = _bitonic_sort(16)
    
    # Merge 17th wire (index 16) into the sorted 16 wires
    # Compare wire i with wire 16 for i = 0 to 15
    for i in range(16):
        network.append((i, 16))
        
    # Verify
    if not _check_network(network, full_check=True):
        # Fallback to OET if something went wrong
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
        
    return network
# EVOLVE-BLOCK-END
