"""Size-efficient sorting networks on 16 wires.
Strategy: start from a known-good 128-comparator Odd-Even Transposition network,
then apply a greedy removal and random shuffle local search to reduce total comparator count.
We verify correctness via the zero-one principle using precomputed bit-vector masks for speed.
"""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096

# Precompute initial masks for all 2^16 inputs
# Mask w has bit k set if input k has 1 on wire w
_INITIAL_MASKS = []
for w in range(N):
    m = 0
    for k in range(1 << N):
        if (k >> w) & 1:
            m |= (1 << k)
    _INITIAL_MASKS.append(m)

def _verify_network_fast(nets):
    """
    Verify network correctness using bit-vector masks.
    """
    # Copy initial masks
    masks = list(_INITIAL_MASKS)
    
    # Apply comparators
    for i, j in nets:
        # S is the set of inputs where wire i is 1 and wire j is 0
        S = masks[i] & ~masks[j]
        if S:
            # Update masks
            masks[i] &= ~S
            masks[j] |= S
            
    # Check sortedness: For all w, masks[w] should be a subset of masks[w+1]
    # i.e., masks[w] & ~masks[w+1] == 0
    for w in range(N - 1):
        if (masks[w] & ~masks[w+1]) != 0:
            return False
    return True

def _odd_even_transposition_sort(n):
    """Generate Odd-Even Transposition Sort network for n wires."""
    nets = []
    for i in range(n):
        for j in range(i % 2, n - 1, 2):
            nets.append((j, j + 1))
    return nets

def construct():
    """Construct a sorting network for 16 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Start with Odd-Even Transposition Sort (128 comparators)
    nets = _odd_even_transposition_sort(N)
    best_nets = list(nets)
    best_count = len(nets)
    
    rng = random.Random(42)
    
    def try_remove_redundant(current_nets):
        """Greedy removal of redundant comparators."""
        removed = True
        while removed:
            removed = False
            # Iterate in random order to escape local minima of removal order
            indices = list(range(len(current_nets)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    return current_nets
                
                candidate = current_nets[:idx] + current_nets[idx+1:]
                if _verify_network_fast(candidate):
                    current_nets = candidate
                    removed = True
                    break # Restart loop with new length
        return current_nets

    # Initial greedy removal
    nets = try_remove_redundant(nets)
    if len(nets) < best_count:
        best_nets = list(nets)
        best_count = len(nets)
        
    # Random shuffle and removal loop
    while time.time() < deadline:
        # Shuffle the current network
        shuffled = list(nets)
        rng.shuffle(shuffled)
        
        # Try to remove redundants from the shuffled version
        candidate = try_remove_redundant(shuffled)
        
        if len(candidate) < best_count:
            best_nets = list(candidate)
            best_count = len(candidate)
            nets = candidate
            
    return best_nets
