"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 80.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _batcher_merge(k, offset):
    """
    Batcher's merge of two sorted sequences of length k.
    The first sequence starts at offset, the second at offset + k.
    Returns list of comparators (i, j) with i < j.
    """
    if k <= 1:
        return []
    
    comps = []
    k2 = k // 2
    
    # Recursive merges
    if k2 > 0:
        comps.extend(_batcher_merge(k2, offset))
        comps.extend(_batcher_merge(k - k2, offset + k2))
    
    # Cross comparisons
    for i in range(k):
        a = offset + i
        b = offset + 2 * k - 1 - i
        if a < b:
            comps.append((a, b))
        else:
            comps.append((b, a))
    
    return comps

def _batcher_sort(n):
    """Generate Batcher's merge sort network for n wires."""
    if n <= 1:
        return []
    
    n1 = n // 2
    n2 = n - n1
    
    comps = []
    # Sort left half
    comps.extend(_batcher_sort(n1))
    # Sort right half (offset by n1)
    right_comps = _batcher_sort(n2)
    comps.extend([(i + n1, j + n1) for i, j in right_comps])
    
    # Merge
    # Batcher merge requires equal halves or specific handling
    # For simplicity, use a standard merge that works for unequal sizes
    # by padding the smaller half
    
    # Actually, let's use a different approach: bitonic sort
    # which is easier to implement correctly for arbitrary n.
    # But let's try to make batcher work.
    
    # For now, if n1 != n2, we need a different merge strategy.
    # Let's just use the standard Batcher merge which assumes equal halves.
    # If n is not a power of 2, we need to be careful.
    
    # Alternative: use bitonic sort
    return _bitonic_sort(n)

def _bitonic_sort(n):
    """Generate a bitonic sort network for n wires."""
    def bitonic_merge(k, offset, direction):
        """Merge a bitonic sequence of length k starting at offset."""
        if k <= 1:
            return []
        
        comps = []
        half = k // 2
        
        # Compare elements half apart
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively merge each half
        # Left half: keep same direction
        # Right half: keep same direction
        comps.extend(bitonic_merge(half, offset, direction))
        if k - half > 0:
            comps.extend(bitonic_merge(k - half, offset + half, direction))
        
        return comps
    
    def bitonic_sort_recursive(k, offset, direction):
        """Sort a bitonic sequence of length k starting at offset."""
        if k <= 1:
            return []
        
        comps = []
        half = k // 2
        
        # Recursively sort each half in opposite directions
        comps.extend(bitonic_sort_recursive(half, offset, direction))
        if k - half > 0:
            comps.extend(bitonic_sort_recursive(k - half, offset + half, -direction))
        
        # Merge
        comps.extend(bitonic_merge(k, offset, direction))
        
        return comps
    
    # For a full sort, we first need to create a bitonic sequence.
    # The standard approach is to sort in a pattern that creates bitonic sequences.
    
    # Simpler: use the standard bitonic sort construction
    # which requires n to be a power of 2 for the classic algorithm.
    # For n=18, we can pad to 32 or use a modified approach.
    
    # Let's use a different strategy: start with odd-even transposition
    # and optimize from there.
    
    return _odd_even_transposition_network(n)

def _verify_network(comparators, all_inputs, sorted_values):
    """Verify that the network sorts all inputs."""
    inputs = all_inputs.copy()
    for (i, j) in comparators:
        mask_i = (inputs >> i) & 1
        mask_j = (inputs >> j) & 1
        swap_mask = mask_j & (~mask_i)
        inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
    
    count = 0
    for s in sorted_values:
        count += np.sum(inputs == s)
    return count == len(inputs)

def _remove_redundant_comparators(network, all_inputs, sorted_values, deadline):
    """Try to remove redundant comparators from the network."""
    if not _verify_network(network, all_inputs, sorted_values):
        return network
    
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            
            candidate = network[:idx] + network[idx + 1:]
            if _verify_network(candidate, all_inputs, sorted_values):
                network = candidate
                improved = True
    
    return network

def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Sorted values: bitmask x is sorted (0s then 1s) iff x = 2^N - 2^k for k in [0, N]
    sorted_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.int64)
    
    # Start with odd-even transposition network (known correct)
    initial_network = _odd_even_transposition_network(N)
    
    # Verify initial network
    if not _verify_network(initial_network, all_inputs, sorted_values):
        # Should not happen, but fallback
        raise ValueError("Initial network is incorrect")
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    # Greedy removal
    network = _remove_redundant_comparators(initial_network, all_inputs, sorted_values, DEADLINE)
    
    if len(network) < best_count:
        best_network = list(network)
        best_count = len(network)
    
    # Try more aggressive optimization with random restarts
    rng = random.Random(42)
    
    # Try different initial networks
    # 1. Odd-even transposition (already done)
    # 2. Try to construct a better network from scratch using known patterns
    
    # For now, focus on optimizing the odd-even transposition network
    
    # Multiple passes of removal with different orderings
    for pass_num in range(3):
        if time.time() > DEADLINE - 5:
            break
        
        network = list(best_network)
        improved = True
        while improved:
            improved = False
            indices = list(range(len(network) - 1, -1, -1))
            if pass_num % 2 == 1:
                indices.reverse()
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > DEADLINE - 5:
                    break
                if idx >= len(network):
                    continue
                
                candidate = network[:idx] + network[idx + 1:]
                if _verify_network(candidate, all_inputs, sorted_values):
                    network = candidate
                    improved = True
                    if len(network) < best_count:
                        best_network = list(network)
                        best_count = len(network)
    
    # Try swapping adjacent comparators to enable further removals
    for iteration in range(10):
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        network = list(best_network)
        
        for idx in range(len(network) - 1):
            if time.time() > DEADLINE - 5:
                break
            
            # Try swapping
            candidate = network[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            
            if _verify_network(candidate, all_inputs, sorted_values):
                # Try to remove comparators from this swapped network
                temp_network = _remove_redundant_comparators(candidate, all_inputs, sorted_values, DEADLINE)
                
                if len(temp_network) < len(network):
                    network = temp_network
                    improved = True
                    if len(network) < best_count:
                        best_network = list(network)
                        best_count = len(network)
        
        if not improved:
            break
    
    return best_network
# EVOLVE-BLOCK-END
