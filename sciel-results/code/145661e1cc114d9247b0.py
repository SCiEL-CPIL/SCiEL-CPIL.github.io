"""Size-efficient sorting networks on 16 wires.
Strategy: start from a known-good 128-comparator Odd-Even Transposition network,
then apply a greedy removal and random shuffle local search to reduce total comparator count.
We verify correctness via the zero-one principle (all 2^16 = 65536 inputs).
"""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096

def _verify_network_fast(nets):
    """Faster verification using numpy vectorization over all inputs at once."""
    # Create array of shape (65536, N) where each row is a binary input
    all_inputs = np.zeros((1 << N, N), dtype=np.int8)
    for k in range(N):
        all_inputs[:, k] = (np.arange(1 << N) >> k) & 1
    
    for (i, j) in nets:
        # Where wire i > wire j, swap
        swap_mask = all_inputs[:, i] > all_inputs[:, j]
        all_inputs[swap_mask, i] = 1 - all_inputs[swap_mask, i]
        all_inputs[swap_mask, j] = 1 - all_inputs[swap_mask, j]
    
    # Check if all rows are sorted
    for k in range(N - 1):
        if np.any(all_inputs[:, k] > all_inputs[:, k + 1]):
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
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
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
            for idx in range(len(current_nets) - 1, -1, -1):
                if time.time() > deadline:
                    return current_nets
                candidate = current_nets[:idx] + current_nets[idx+1:]
                if _verify_network_fast(candidate):
                    current_nets = candidate
                    removed = True
        return current_nets

    # Initial greedy removal
    nets = try_remove_redundant(nets)
    if len(nets) < best_count:
        best_nets = list(nets)
        best_count = len(nets)
        
    # Random shuffle and removal
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Shuffle the current network
        shuffled = list(nets)
        rng.shuffle(shuffled)
        
        # Try to remove redundants from the shuffled version
        candidate = try_remove_redundant(shuffled)
        
        if len(candidate) < best_count:
            best_nets = list(candidate)
            best_count = len(candidate)
            nets = candidate
        else:
            # If we didn't improve, keep the previous best for the next shuffle base?
            # Or keep the current nets?
            # Keeping the current nets (which is already minimized) is better.
            pass
            
    return best_nets
